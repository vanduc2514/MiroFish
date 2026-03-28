"""
Local Graphiti + Neo4j graph provider implementation.
"""

from __future__ import annotations

import atexit
import asyncio
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from ...config import Config
from ...utils.logger import get_logger
from ...utils.ontology_normalizer import normalize_ontology_for_zep
from .base import BaseGraphProvider, ProgressCallback
from .models import GraphEdgeRecord, GraphNodeRecord, GraphSearchResult

logger = get_logger('mirofish.graph_provider.graphiti_local')


class _AsyncRunner:
    """Run all Graphiti/Neo4j async work on one dedicated event loop thread."""

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="graphiti-local-loop", daemon=True)
        self._started = threading.Event()
        self._closed = False
        self._thread.start()
        self._started.wait()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()

    def run(self, coro):
        if self._closed:
            raise RuntimeError("Async runner is already closed")

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.close()


_ASYNC_RUNNER = _AsyncRunner()
atexit.register(_ASYNC_RUNNER.close)


def _run_async(coro):
    return _ASYNC_RUNNER.run(coro)


@dataclass
class _OntologyBundle:
    entity_types: dict[str, type[BaseModel]]
    edge_types: dict[str, type[BaseModel]]
    edge_type_map: dict[tuple[str, str], list[str]]
    attribute_free_entity_types: dict[str, type[BaseModel]]
    attribute_free_edge_types: dict[str, type[BaseModel]]


class GraphitiLocalGraphProvider(BaseGraphProvider):
    """Graphiti + Neo4j backed graph provider."""

    _initialized = False
    # Startup can flow through ensure_initialized() -> _ensure_client_ready(), so this
    # lock must be re-entrant to avoid self-deadlocking during app bootstrap.
    _init_lock = threading.RLock()

    def __init__(self):
        try:
            from graphiti_core import Graphiti
            from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
            from graphiti_core.driver.neo4j_driver import Neo4jDriver
            from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
            from graphiti_core.errors import GroupsEdgesNotFoundError, GroupsNodesNotFoundError, NodeNotFoundError
            from graphiti_core.llm_client.config import LLMConfig
            from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
            from neo4j.exceptions import ClientError
        except ImportError as exc:  # pragma: no cover - depends on installed extras
            raise ImportError(
                "graphiti-core and neo4j must be installed to use GRAPH_BACKEND=graphiti_local"
            ) from exc

        self._Graphiti = Graphiti
        self._Neo4jDriver = Neo4jDriver
        self._OpenAIEmbedder = OpenAIEmbedder
        self._OpenAIEmbedderConfig = OpenAIEmbedderConfig
        self._OpenAIRerankerClient = OpenAIRerankerClient
        self._OpenAIGenericClient = OpenAIGenericClient
        self._LLMConfig = LLMConfig
        self._GroupsEdgesNotFoundError = GroupsEdgesNotFoundError
        self._GroupsNodesNotFoundError = GroupsNodesNotFoundError
        self._NodeNotFoundError = NodeNotFoundError
        self._ClientError = ClientError

        # Graphiti reads this env var directly.
        os.environ.setdefault('GRAPHITI_TELEMETRY_ENABLED', str(Config.GRAPHITI_TELEMETRY_ENABLED).lower())

        self._llm_config = self._LLMConfig(
            api_key=Config.GRAPHITI_LLM_API_KEY,
            base_url=Config.GRAPHITI_LLM_BASE_URL,
            model=Config.GRAPHITI_LLM_MODEL,
        )
        self._reranker_config = self._LLMConfig(
            api_key=Config.GRAPHITI_RERANKER_API_KEY,
            base_url=Config.GRAPHITI_RERANKER_BASE_URL,
            model=Config.GRAPHITI_RERANKER_MODEL,
        )
        self._embedder_config = self._OpenAIEmbedderConfig(
            api_key=Config.GRAPHITI_EMBEDDER_API_KEY,
            base_url=Config.GRAPHITI_EMBEDDER_BASE_URL,
            embedding_model=Config.GRAPHITI_EMBEDDER_MODEL,
        )

        self.driver = self._Neo4jDriver(
            uri=Config.NEO4J_URI,
            user=Config.NEO4J_USER,
            password=Config.NEO4J_PASSWORD,
            database=Config.NEO4J_DATABASE,
        )
        self.client = self._make_graphiti_client(self.driver)
        self._ontology_cache: dict[str, _OntologyBundle] = {}
        self._client_ready = False

    def ensure_initialized(self) -> None:
        if GraphitiLocalGraphProvider._initialized or not Config.GRAPHITI_AUTO_INIT:
            return

        with GraphitiLocalGraphProvider._init_lock:
            if GraphitiLocalGraphProvider._initialized:
                return
            self._ensure_client_ready()
            GraphitiLocalGraphProvider._initialized = True

    def _ensure_client_ready(self) -> None:
        if self._client_ready:
            return

        with GraphitiLocalGraphProvider._init_lock:
            if self._client_ready:
                return
            logger.info("Checking local Neo4j connectivity...")
            _run_async(self.driver.health_check())
            logger.info("Local Neo4j connectivity confirmed")
            logger.info("Initializing local Graphiti indices and constraints...")
            _run_async(self.client.build_indices_and_constraints())
            self._client_ready = True
            logger.info("Local Graphiti initialization completed")

    def create_graph(self, name: str) -> str:
        self._ensure_client_ready()
        graph_id = f"mirofish_{uuid.uuid4().hex[:16]}"
        logger.info("Created local Graphiti graph namespace %s (%s)", graph_id, name)
        return graph_id

    def set_ontology(self, graph_id: str, ontology: dict[str, Any]) -> None:
        self._ontology_cache[graph_id] = self._build_ontology_bundle(ontology)

    def add_text_batches(
        self,
        graph_id: str,
        chunks: list[str],
        batch_size: int = 3,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> list[str]:
        self._ensure_client_ready()
        client = self._get_graphiti_client(graph_id)
        bundle = self._ontology_cache.get(graph_id)
        episode_uuids: list[str] = []
        total_chunks = len(chunks)

        from graphiti_core.nodes import EpisodeType

        base_time = datetime.now(timezone.utc)

        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size

            if progress_callback:
                progress_callback(
                    f"Sending local batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)...",
                    (i + len(batch_chunks)) / total_chunks if total_chunks else 1.0,
                )

            for index, chunk in enumerate(batch_chunks):
                result = _run_async(
                    self._add_episode(
                        client=client,
                        graph_id=graph_id,
                        name=f"{graph_id}_chunk_{i + index + 1}",
                        episode_body=chunk,
                        source_description="MiroFish document chunk",
                        reference_time=base_time + timedelta(seconds=i + index),
                        source=EpisodeType.text,
                        bundle=bundle,
                    )
                )
                self._persist_graph_result(client, result)

                episode = getattr(result, 'episode', None)
                episode_uuid = getattr(episode, 'uuid', None) or getattr(episode, 'uuid_', None)
                if episode_uuid:
                    episode_uuids.append(str(episode_uuid))

        return episode_uuids

    def wait_for_episodes(
        self,
        graph_id: str,
        episode_uuids: list[str],
        progress_callback: Optional[ProgressCallback] = None,
        timeout: int = 600,
    ) -> None:
        if progress_callback:
            progress_callback(
                "Local Graphiti ingestion completed",
                1.0,
            )

    def get_all_nodes(self, graph_id: str) -> list[GraphNodeRecord]:
        self._ensure_client_ready()
        from graphiti_core.nodes import EntityNode

        return [
            self._normalize_node(node)
            for node in self._fetch_group_records(EntityNode.get_by_group_ids, graph_id)
        ]

    def get_all_edges(self, graph_id: str) -> list[GraphEdgeRecord]:
        self._ensure_client_ready()
        from graphiti_core.edges import EntityEdge

        return [
            self._normalize_edge(edge)
            for edge in self._fetch_group_records(EntityEdge.get_by_group_ids, graph_id)
        ]

    def get_node(self, graph_id: str, node_uuid: str) -> GraphNodeRecord | None:
        self._ensure_client_ready()
        from graphiti_core.nodes import EntityNode

        graph_driver = self._get_graph_driver(graph_id)
        try:
            node = _run_async(EntityNode.get_by_uuid(graph_driver, node_uuid))
        except self._NodeNotFoundError:
            return None

        if graph_id and getattr(node, 'group_id', None) not in (None, *self._graph_namespaces(graph_id)):
            return None
        return self._normalize_node(node)

    def get_node_edges(self, graph_id: str, node_uuid: str) -> list[GraphEdgeRecord]:
        self._ensure_client_ready()
        from graphiti_core.edges import EntityEdge

        graph_driver = self._get_graph_driver(graph_id)
        edges = _run_async(EntityEdge.get_by_node_uuid(graph_driver, node_uuid))
        return [
            self._normalize_edge(edge)
            for edge in edges
            if not graph_id or getattr(edge, 'group_id', None) in (None, *self._graph_namespaces(graph_id))
        ]

    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
        reranker: str = "cross_encoder",
    ) -> GraphSearchResult:
        self._ensure_client_ready()
        client = self._get_graphiti_client(graph_id)
        from graphiti_core.search.search_config_recipes import (
            EDGE_HYBRID_SEARCH_CROSS_ENCODER,
            EDGE_HYBRID_SEARCH_RRF,
            NODE_HYBRID_SEARCH_CROSS_ENCODER,
            NODE_HYBRID_SEARCH_RRF,
        )

        effective_reranker = Config.GRAPHITI_SEARCH_RERANKER or reranker or "rrf"

        if scope == "nodes":
            config = (
                NODE_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
                if effective_reranker == "cross_encoder"
                else NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
            )
        else:
            config = (
                EDGE_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
                if effective_reranker == "cross_encoder"
                else EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
            )
        config.limit = limit

        results = _run_async(
            client.search_(
                query=query,
                config=config,
                group_ids=self._graph_namespaces(graph_id),
            )
        )

        edges = [self._normalize_edge(edge) for edge in results.edges]
        nodes = [self._normalize_node(node) for node in results.nodes]

        facts = [edge.fact for edge in edges if edge.fact]
        if scope == "nodes":
            facts.extend(f"[{node.name}]: {node.summary}" for node in nodes if node.summary)

        return GraphSearchResult(facts=facts, edges=edges, nodes=nodes)

    def add_text(
        self,
        graph_id: str,
        data: str,
        source_description: str = "MiroFish",
    ) -> str | None:
        self._ensure_client_ready()
        client = self._get_graphiti_client(graph_id)
        from graphiti_core.nodes import EpisodeType

        result = _run_async(
            self._add_episode(
                client=client,
                graph_id=graph_id,
                name=f"{graph_id}_activity_{uuid.uuid4().hex[:8]}",
                episode_body=data,
                source_description=source_description,
                reference_time=datetime.now(timezone.utc),
                source=EpisodeType.text,
            )
        )
        self._persist_graph_result(client, result)
        episode = getattr(result, 'episode', None)
        episode_uuid = getattr(episode, 'uuid', None) if episode else None
        return str(episode_uuid) if episode_uuid else None

    def delete_graph(self, graph_id: str) -> None:
        self._ensure_client_ready()
        from graphiti_core.edges import EntityEdge, EpisodicEdge
        from graphiti_core.nodes import EntityNode, EpisodicNode

        graph_driver = self._get_graph_driver(graph_id)
        entity_edges = self._fetch_group_records(EntityEdge.get_by_group_ids, graph_id)
        episodic_edges = self._fetch_group_records(EpisodicEdge.get_by_group_ids, graph_id)
        episodic_nodes = self._fetch_group_records(EpisodicNode.get_by_group_ids, graph_id)
        entity_nodes = self._fetch_group_records(EntityNode.get_by_group_ids, graph_id)

        if episodic_edges:
            _run_async(EpisodicEdge.delete_by_uuids(graph_driver, [edge.uuid for edge in episodic_edges]))
        if entity_edges:
            _run_async(EntityEdge.delete_by_uuids(graph_driver, [edge.uuid for edge in entity_edges]))
        if episodic_nodes:
            _run_async(EpisodicNode.delete_by_uuids(graph_driver, [node.uuid for node in episodic_nodes]))
        if entity_nodes:
            _run_async(EntityNode.delete_by_uuids(graph_driver, [node.uuid for node in entity_nodes]))

        self._ontology_cache.pop(graph_id, None)

    def _fetch_group_records(self, fetcher, graph_id: str, page_size: int = 100) -> list[Any]:
        graph_driver = self._get_graph_driver(graph_id)
        graph_namespaces = self._graph_namespaces(graph_id)
        records: list[Any] = []
        cursor: str | None = None

        while True:
            try:
                batch = _run_async(
                    fetcher(
                        graph_driver,
                        graph_namespaces,
                        limit=page_size,
                        uuid_cursor=cursor,
                    )
                )
            except (self._GroupsEdgesNotFoundError, self._GroupsNodesNotFoundError):
                break
            if not batch:
                break

            records.extend(batch)
            if len(batch) < page_size:
                break

            cursor = getattr(batch[-1], 'uuid', None) or getattr(batch[-1], 'uuid_', None)
            if cursor is None:
                break

        return records

    def _graph_namespace(self, graph_id: str) -> str:
        if not graph_id or not re.fullmatch(r'[A-Za-z0-9_-]+', graph_id):
            raise ValueError(f"Invalid graph_id for local Graphiti backend: {graph_id}")
        return graph_id

    def _graph_namespaces(self, graph_id: str) -> list[str]:
        primary = self._graph_namespace(graph_id)
        namespaces = [primary]
        legacy = primary.replace('_', '-')
        if legacy != primary:
            namespaces.append(legacy)
        return namespaces

    def _make_graphiti_client(self, graph_driver) -> Any:
        return self._Graphiti(
            graph_driver=graph_driver,
            llm_client=self._OpenAIGenericClient(config=self._llm_config),
            embedder=self._OpenAIEmbedder(config=self._embedder_config),
            cross_encoder=self._OpenAIRerankerClient(config=self._reranker_config),
            max_coroutines=Config.GRAPHITI_MAX_COROUTINES,
        )

    def _get_graphiti_client(self, graph_id: str):
        self._graph_namespace(graph_id)
        self._ensure_client_ready()
        return self.client

    def _get_graph_driver(self, graph_id: str):
        return self._get_graphiti_client(graph_id).driver

    async def _add_episode(
        self,
        client,
        graph_id: str,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
        source,
        bundle: _OntologyBundle | None = None,
    ):
        episode_kwargs = {
            "name": name,
            "episode_body": episode_body,
            "source_description": source_description,
            "reference_time": reference_time,
            "source": source,
            "group_id": self._graph_namespace(graph_id),
            "entity_types": bundle.entity_types if bundle else None,
            "edge_types": bundle.edge_types if bundle else None,
            "edge_type_map": bundle.edge_type_map if bundle else None,
        }

        try:
            return await client.add_episode(**episode_kwargs)
        except Exception as exc:
            if not bundle or not self._is_non_primitive_property_error(exc):
                raise

            logger.warning(
                "Local Graphiti ontology extraction returned non-primitive Neo4j properties for %s; retrying without ontology attributes. Error: %s",
                graph_id,
                exc,
            )
            fallback_kwargs = dict(episode_kwargs)
            fallback_kwargs.update(
                entity_types=bundle.attribute_free_entity_types,
                edge_types=bundle.attribute_free_edge_types,
                edge_type_map=bundle.edge_type_map,
            )
            return await client.add_episode(**fallback_kwargs)

    def _persist_graph_result(self, client, result: Any) -> None:
        for node in getattr(result, 'nodes', []) or []:
            node.attributes = self._sanitize_attributes(getattr(node, 'attributes', {}) or {})
            if getattr(node, 'name_embedding', None) is None:
                _run_async(node.generate_name_embedding(client.embedder))
            _run_async(node.save(client.driver))

        for edge in getattr(result, 'edges', []) or []:
            edge.attributes = self._sanitize_attributes(getattr(edge, 'attributes', {}) or {})
            if getattr(edge, 'fact_embedding', None) is None:
                _run_async(edge.generate_embedding(client.embedder))
            _run_async(edge.save(client.driver))

    @staticmethod
    def _is_non_primitive_property_error(exc: Exception) -> bool:
        return 'Property values can only be of primitive types or arrays thereof' in str(exc)

    def _sanitize_attributes(self, attributes: dict[str, Any]) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}
        for key, value in attributes.items():
            sanitized[key] = self._sanitize_property_value(key, value)
        return sanitized

    def _sanitize_property_value(self, key: str, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, (list, tuple)):
            return [
                item
                if isinstance(item, (str, int, float, bool)) or item is None
                else json.dumps(item, ensure_ascii=False, default=str)
                for item in value
            ]

        if isinstance(value, dict):
            if key in value:
                return self._sanitize_property_value(key, value[key])
            if len(value) == 1:
                return self._sanitize_property_value(key, next(iter(value.values())))
            return json.dumps(value, ensure_ascii=False, default=str)

        return str(value)

    def _build_ontology_bundle(self, ontology: dict[str, Any]) -> _OntologyBundle:
        ontology, _ = normalize_ontology_for_zep(ontology)
        reserved_names = {
            'uuid',
            'name',
            'group_id',
            'labels',
            'created_at',
            'summary',
            'attributes',
            'name_embedding',
        }

        def safe_attr_name(attr_name: str) -> str:
            if attr_name.lower() in reserved_names:
                return f"entity_{attr_name}"
            return attr_name

        entity_types: dict[str, type[BaseModel]] = {}
        for entity_def in ontology.get("entity_types", []):
            name = entity_def["name"]
            description = entity_def.get("description", f"A {name} entity.")
            attrs: dict[str, Any] = {"__doc__": description}
            annotations: dict[str, Any] = {}
            for attr_def in entity_def.get("attributes", []):
                attr_name = safe_attr_name(attr_def["name"])
                attr_desc = attr_def.get("description", attr_name)
                attrs[attr_name] = Field(default=None, description=attr_desc)
                annotations[attr_name] = Optional[str]
            attrs["__annotations__"] = annotations
            entity_class = type(name, (BaseModel,), attrs)
            entity_class.__doc__ = description
            entity_types[name] = entity_class

        edge_types: dict[str, type[BaseModel]] = {}
        edge_type_map: dict[tuple[str, str], list[str]] = {}
        for edge_def in ontology.get("edge_types", []):
            name = edge_def["name"]
            description = edge_def.get("description", f"A {name} relationship.")
            attrs = {"__doc__": description}
            annotations = {}
            for attr_def in edge_def.get("attributes", []):
                attr_name = safe_attr_name(attr_def["name"])
                attr_desc = attr_def.get("description", attr_name)
                attrs[attr_name] = Field(default=None, description=attr_desc)
                annotations[attr_name] = Optional[str]
            attrs["__annotations__"] = annotations
            edge_class = type(name, (BaseModel,), attrs)
            edge_class.__doc__ = description
            edge_types[name] = edge_class

            source_targets = edge_def.get("source_targets", []) or [{"source": "Entity", "target": "Entity"}]
            for source_target in source_targets:
                signature = (
                    source_target.get("source", "Entity"),
                    source_target.get("target", "Entity"),
                )
                edge_type_map.setdefault(signature, []).append(name)

        return _OntologyBundle(
            entity_types=entity_types,
            edge_types=edge_types,
            edge_type_map=edge_type_map,
            attribute_free_entity_types=self._build_attribute_free_models(entity_types),
            attribute_free_edge_types=self._build_attribute_free_models(edge_types),
        )

    @staticmethod
    def _build_attribute_free_models(
        typed_models: dict[str, type[BaseModel]]
    ) -> dict[str, type[BaseModel]]:
        stripped_models: dict[str, type[BaseModel]] = {}
        for model_name, model_type in typed_models.items():
            attrs: dict[str, Any] = {
                "__doc__": model_type.__doc__ or f"A {model_name} type.",
                "__annotations__": {},
            }
            stripped_model = type(model_name, (BaseModel,), attrs)
            stripped_model.__doc__ = model_type.__doc__
            stripped_models[model_name] = stripped_model
        return stripped_models

    @staticmethod
    def _normalize_node(node: Any) -> GraphNodeRecord:
        created_at = getattr(node, 'created_at', None)
        return GraphNodeRecord(
            uuid=str(getattr(node, 'uuid', None) or getattr(node, 'uuid_', None) or ""),
            name=getattr(node, 'name', '') or "",
            labels=getattr(node, 'labels', []) or [],
            summary=getattr(node, 'summary', '') or "",
            attributes=getattr(node, 'attributes', {}) or {},
            created_at=str(created_at) if created_at else None,
        )

    @staticmethod
    def _normalize_edge(edge: Any) -> GraphEdgeRecord:
        episodes = getattr(edge, 'episodes', None) or []
        if not isinstance(episodes, list):
            episodes = [str(episodes)]
        return GraphEdgeRecord(
            uuid=str(getattr(edge, 'uuid', None) or getattr(edge, 'uuid_', None) or ""),
            name=getattr(edge, 'name', '') or "",
            fact=getattr(edge, 'fact', '') or "",
            source_node_uuid=getattr(edge, 'source_node_uuid', '') or "",
            target_node_uuid=getattr(edge, 'target_node_uuid', '') or "",
            attributes=getattr(edge, 'attributes', {}) or {},
            created_at=str(getattr(edge, 'created_at', None)) if getattr(edge, 'created_at', None) else None,
            valid_at=str(getattr(edge, 'valid_at', None)) if getattr(edge, 'valid_at', None) else None,
            invalid_at=str(getattr(edge, 'invalid_at', None)) if getattr(edge, 'invalid_at', None) else None,
            expired_at=str(getattr(edge, 'expired_at', None)) if getattr(edge, 'expired_at', None) else None,
            episodes=[str(episode) for episode in episodes],
        )

"""
Zep Cloud graph provider implementation.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from zep_cloud import EpisodeData, EntityEdgeSourceTarget
from zep_cloud.client import Zep

from ...config import Config
from ...utils.logger import get_logger
from ...utils.ontology_normalizer import normalize_ontology_for_zep
from ...utils.zep_paging import fetch_all_edges, fetch_all_nodes
from .base import BaseGraphProvider, ProgressCallback
from .models import GraphEdgeRecord, GraphNodeRecord, GraphSearchResult

logger = get_logger('mirofish.graph_provider.zep_cloud')


class ZepCloudGraphProvider(BaseGraphProvider):
    """Zep Cloud backed graph provider."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.ZEP_API_KEY
        if not self.api_key:
            raise ValueError("ZEP_API_KEY 未配置")

        self.client = Zep(api_key=self.api_key)

    def create_graph(self, name: str) -> str:
        graph_id = f"mirofish_{uuid.uuid4().hex[:16]}"
        self.client.graph.create(
            graph_id=graph_id,
            name=name,
            description="MiroFish Social Simulation Graph",
        )
        return graph_id

    def set_ontology(self, graph_id: str, ontology: dict[str, Any]) -> None:
        import warnings
        from pydantic import Field
        from zep_cloud.external_clients.ontology import EdgeModel, EntityModel, EntityText

        warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

        ontology, entity_name_mapping = normalize_ontology_for_zep(ontology)
        renamed_entities = {
            original: normalized
            for original, normalized in entity_name_mapping.items()
            if original != normalized
        }
        if renamed_entities:
            logger.info("Normalized ontology entity names for Zep compatibility: %s", renamed_entities)

        reserved_names = {'uuid', 'name', 'group_id', 'name_embedding', 'summary', 'created_at'}

        def safe_attr_name(attr_name: str) -> str:
            if attr_name.lower() in reserved_names:
                return f"entity_{attr_name}"
            return attr_name

        entity_types: dict[str, type[EntityModel]] = {}
        for entity_def in ontology.get("entity_types", []):
            name = entity_def["name"]
            description = entity_def.get("description", f"A {name} entity.")
            attrs: dict[str, Any] = {"__doc__": description}
            annotations: dict[str, Any] = {}

            for attr_def in entity_def.get("attributes", []):
                attr_name = safe_attr_name(attr_def["name"])
                attr_desc = attr_def.get("description", attr_name)
                attrs[attr_name] = Field(description=attr_desc, default=None)
                annotations[attr_name] = Optional[EntityText]

            attrs["__annotations__"] = annotations
            entity_class = type(name, (EntityModel,), attrs)
            entity_class.__doc__ = description
            entity_types[name] = entity_class

        edge_definitions = {}
        for edge_def in ontology.get("edge_types", []):
            name = edge_def["name"]
            description = edge_def.get("description", f"A {name} relationship.")
            attrs = {"__doc__": description}
            annotations = {}

            for attr_def in edge_def.get("attributes", []):
                attr_name = safe_attr_name(attr_def["name"])
                attr_desc = attr_def.get("description", attr_name)
                attrs[attr_name] = Field(description=attr_desc, default=None)
                annotations[attr_name] = Optional[str]

            attrs["__annotations__"] = annotations
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            edge_class = type(class_name, (EdgeModel,), attrs)
            edge_class.__doc__ = description

            source_targets = []
            for st in edge_def.get("source_targets", []):
                source_targets.append(
                    EntityEdgeSourceTarget(
                        source=st.get("source", "Entity"),
                        target=st.get("target", "Entity"),
                    )
                )

            if source_targets:
                edge_definitions[name] = (edge_class, source_targets)

        if entity_types or edge_definitions:
            self.client.graph.set_ontology(
                graph_ids=[graph_id],
                entities=entity_types if entity_types else None,
                edges=edge_definitions if edge_definitions else None,
            )

    def add_text_batches(
        self,
        graph_id: str,
        chunks: list[str],
        batch_size: int = 3,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> list[str]:
        episode_uuids: list[str] = []
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size

            if progress_callback:
                progress_callback(
                    f"发送第 {batch_num}/{total_batches} 批数据 ({len(batch_chunks)} 块)...",
                    (i + len(batch_chunks)) / total_chunks if total_chunks else 1.0,
                )

            episodes = [EpisodeData(data=chunk, type="text") for chunk in batch_chunks]
            batch_result = self.client.graph.add_batch(graph_id=graph_id, episodes=episodes)

            if batch_result and isinstance(batch_result, list):
                for episode in batch_result:
                    episode_uuid = getattr(episode, 'uuid_', None) or getattr(episode, 'uuid', None)
                    if episode_uuid:
                        episode_uuids.append(str(episode_uuid))

            time.sleep(1)

        return episode_uuids

    def wait_for_episodes(
        self,
        graph_id: str,
        episode_uuids: list[str],
        progress_callback: Optional[ProgressCallback] = None,
        timeout: int = 600,
    ) -> None:
        if not episode_uuids:
            if progress_callback:
                progress_callback("无需等待（没有 episode）", 1.0)
            return

        start_time = time.time()
        pending_episodes = set(episode_uuids)
        completed_count = 0
        total_episodes = len(episode_uuids)

        if progress_callback:
            progress_callback(f"开始等待 {total_episodes} 个文本块处理...", 0)

        while pending_episodes:
            if time.time() - start_time > timeout:
                if progress_callback:
                    progress_callback(
                        f"部分文本块超时，已完成 {completed_count}/{total_episodes}",
                        completed_count / total_episodes if total_episodes else 1.0,
                    )
                break

            for episode_uuid in list(pending_episodes):
                try:
                    episode = self.client.graph.episode.get(uuid_=episode_uuid)
                except Exception:
                    continue

                if getattr(episode, 'processed', False):
                    pending_episodes.remove(episode_uuid)
                    completed_count += 1

            if progress_callback:
                elapsed = int(time.time() - start_time)
                progress_callback(
                    f"Zep处理中... {completed_count}/{total_episodes} 完成, {len(pending_episodes)} 待处理 ({elapsed}秒)",
                    completed_count / total_episodes if total_episodes else 1.0,
                )

            if pending_episodes:
                time.sleep(3)

        if progress_callback:
            progress_callback(f"处理完成: {completed_count}/{total_episodes}", 1.0)

    def get_all_nodes(self, graph_id: str) -> list[GraphNodeRecord]:
        return [self._normalize_node(node) for node in fetch_all_nodes(self.client, graph_id)]

    def get_all_edges(self, graph_id: str) -> list[GraphEdgeRecord]:
        return [self._normalize_edge(edge) for edge in fetch_all_edges(self.client, graph_id)]

    def get_node(self, graph_id: str, node_uuid: str) -> GraphNodeRecord | None:
        node = self.client.graph.node.get(uuid_=node_uuid)
        return self._normalize_node(node) if node else None

    def get_node_edges(self, graph_id: str, node_uuid: str) -> list[GraphEdgeRecord]:
        edges = self.client.graph.node.get_entity_edges(node_uuid=node_uuid)
        return [self._normalize_edge(edge) for edge in edges]

    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
        reranker: str = "cross_encoder",
    ) -> GraphSearchResult:
        search_results = self.client.graph.search(
            graph_id=graph_id,
            query=query,
            limit=limit,
            scope=scope,
            reranker=reranker,
        )

        edges = [
            self._normalize_edge(edge)
            for edge in getattr(search_results, 'edges', []) or []
        ]
        nodes = [
            self._normalize_node(node)
            for node in getattr(search_results, 'nodes', []) or []
        ]

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
        result = self.client.graph.add(graph_id=graph_id, type="text", data=data)
        episode_uuid = getattr(result, 'uuid_', None) or getattr(result, 'uuid', None)
        return str(episode_uuid) if episode_uuid else None

    def delete_graph(self, graph_id: str) -> None:
        self.client.graph.delete(graph_id=graph_id)

    @staticmethod
    def _normalize_node(node: Any) -> GraphNodeRecord:
        node_uuid = getattr(node, 'uuid_', None) or getattr(node, 'uuid', None) or ""
        created_at = getattr(node, 'created_at', None)
        return GraphNodeRecord(
            uuid=str(node_uuid),
            name=getattr(node, 'name', '') or "",
            labels=getattr(node, 'labels', []) or [],
            summary=getattr(node, 'summary', '') or "",
            attributes=getattr(node, 'attributes', {}) or {},
            created_at=str(created_at) if created_at else None,
        )

    @staticmethod
    def _normalize_edge(edge: Any) -> GraphEdgeRecord:
        edge_uuid = getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', None) or ""
        episodes = getattr(edge, 'episodes', None) or getattr(edge, 'episode_ids', None) or []
        if not isinstance(episodes, list):
            episodes = [str(episodes)]
        return GraphEdgeRecord(
            uuid=str(edge_uuid),
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

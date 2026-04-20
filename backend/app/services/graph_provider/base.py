"""
Abstract graph provider interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Optional

from .models import GraphEdgeRecord, GraphNodeRecord, GraphSearchResult

ProgressCallback = Callable[[str, float], None]


class BaseGraphProvider(ABC):
    """Provider-neutral graph backend interface."""

    def ensure_initialized(self) -> None:
        """Perform one-time backend initialization when needed."""

    @abstractmethod
    def create_graph(self, name: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def set_ontology(self, graph_id: str, ontology: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_text_batches(
        self,
        graph_id: str,
        chunks: list[str],
        batch_size: int = 3,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def wait_for_episodes(
        self,
        graph_id: str,
        episode_uuids: list[str],
        progress_callback: Optional[ProgressCallback] = None,
        timeout: int = 600,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_all_nodes(self, graph_id: str) -> list[GraphNodeRecord]:
        raise NotImplementedError

    @abstractmethod
    def get_all_edges(self, graph_id: str) -> list[GraphEdgeRecord]:
        raise NotImplementedError

    @abstractmethod
    def get_node(self, graph_id: str, node_uuid: str) -> GraphNodeRecord | None:
        raise NotImplementedError

    @abstractmethod
    def get_node_edges(self, graph_id: str, node_uuid: str) -> list[GraphEdgeRecord]:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
        reranker: str = "cross_encoder",
    ) -> GraphSearchResult:
        raise NotImplementedError

    @abstractmethod
    def add_text(
        self,
        graph_id: str,
        data: str,
        source_description: str = "MiroFish",
    ) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def delete_graph(self, graph_id: str) -> None:
        raise NotImplementedError

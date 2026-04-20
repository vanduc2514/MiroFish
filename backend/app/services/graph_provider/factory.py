"""
Graph provider factory and backend bootstrap helpers.
"""

from __future__ import annotations

from functools import lru_cache

from ...config import Config


@lru_cache(maxsize=2)
def _create_graph_provider_for_backend(backend: str):
    if backend == "zep_cloud":
        from .zep_cloud_provider import ZepCloudGraphProvider

        return ZepCloudGraphProvider()

    if backend == "graphiti_local":
        from .graphiti_local_provider import GraphitiLocalGraphProvider

        return GraphitiLocalGraphProvider()

    raise ValueError(f"Unsupported GRAPH_BACKEND: {backend}")


def create_graph_provider():
    return _create_graph_provider_for_backend(Config.GRAPH_BACKEND)


def initialize_selected_graph_backend() -> None:
    provider = create_graph_provider()
    provider.ensure_initialized()

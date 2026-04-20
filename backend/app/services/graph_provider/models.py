"""
Provider-neutral graph data models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GraphNodeRecord:
    uuid: str
    name: str
    labels: List[str] = field(default_factory=list)
    summary: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None


@dataclass
class GraphEdgeRecord:
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    episodes: List[str] = field(default_factory=list)


@dataclass
class GraphSearchResult:
    facts: List[str] = field(default_factory=list)
    edges: List[GraphEdgeRecord] = field(default_factory=list)
    nodes: List[GraphNodeRecord] = field(default_factory=list)

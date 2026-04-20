"""
Utilities for normalizing ontology names before sending them to Zep.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, Tuple


PASCAL_CASE_PATTERN = re.compile(r"^[A-Z][A-Za-z0-9]*$")


def _split_name_parts(raw_name: str) -> list[str]:
    text = str(raw_name or "").strip()
    if not text:
        return []

    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])(?=[0-9])", " ", text)
    text = re.sub(r"(?<=[0-9])(?=[A-Za-z])", " ", text)
    return [part for part in text.split() if part]


def normalize_pascal_case_name(raw_name: str, default_prefix: str = "Entity") -> str:
    """
    Convert an arbitrary label into Zep-safe PascalCase.
    """
    text = str(raw_name or "").strip()
    if text and PASCAL_CASE_PATTERN.match(text):
        return text

    parts = _split_name_parts(text)
    if not parts:
        return default_prefix

    normalized_parts = []
    for part in parts:
        if part.isdigit():
            normalized_parts.append(part)
        elif part.isupper() and len(part) > 1:
            normalized_parts.append(part)
        else:
            normalized_parts.append(part[0].upper() + part[1:].lower())

    normalized = "".join(normalized_parts)

    if not normalized:
        normalized = default_prefix
    elif not normalized[0].isalpha():
        normalized = f"{default_prefix}{normalized}"

    return normalized


def _ensure_unique_name(base_name: str, used_names: set[str]) -> str:
    candidate = base_name
    suffix = 2

    while candidate in used_names:
        candidate = f"{base_name}{suffix}"
        suffix += 1

    used_names.add(candidate)
    return candidate


def normalize_ontology_for_zep(ontology: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Normalize ontology entity names and source/target references for Zep validation.

    Returns:
        A tuple of (normalized_ontology, entity_name_mapping)
    """
    normalized = copy.deepcopy(ontology or {})
    entity_types = normalized.setdefault("entity_types", [])
    edge_types = normalized.setdefault("edge_types", [])

    used_entity_names: set[str] = set()
    entity_name_mapping: Dict[str, str] = {}

    for entity in entity_types:
        raw_name = str(entity.get("name", "")).strip()
        safe_name = normalize_pascal_case_name(raw_name, default_prefix="Entity")
        safe_name = _ensure_unique_name(safe_name, used_entity_names)

        entity["name"] = safe_name

        if raw_name:
            entity_name_mapping[raw_name] = safe_name
            entity_name_mapping[raw_name.strip()] = safe_name
        entity_name_mapping[safe_name] = safe_name

    for edge in edge_types:
        source_targets = edge.setdefault("source_targets", [])
        for source_target in source_targets:
            raw_source = str(source_target.get("source", "")).strip()
            raw_target = str(source_target.get("target", "")).strip()

            if raw_source:
                source_target["source"] = entity_name_mapping.get(
                    raw_source,
                    normalize_pascal_case_name(raw_source, default_prefix="Entity"),
                )
            else:
                source_target["source"] = "Entity"

            if raw_target:
                source_target["target"] = entity_name_mapping.get(
                    raw_target,
                    normalize_pascal_case_name(raw_target, default_prefix="Entity"),
                )
            else:
                source_target["target"] = "Entity"

    return normalized, entity_name_mapping

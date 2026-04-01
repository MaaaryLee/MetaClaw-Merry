from __future__ import annotations

import re
from typing import Any


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_][a-zA-Z0-9_-]*")


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(text)}


def overlap_coefficient(left: str, right: str) -> float:
    """Cheap lexical overlap proxy for advisor-demo attribution analysis."""
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens) / float(min(len(left_tokens), len(right_tokens)))
    return round(overlap, 4)


def serialize_skill(skill: dict[str, Any], rank: int) -> dict[str, Any]:
    name = str(skill.get("name", "") or "")
    description = str(skill.get("description", "") or "")
    content = str(skill.get("content", "") or "")
    text = " ".join(part for part in [name, description, content] if part).strip()
    return {
        "kind": "skill",
        "skill_name": name,
        "skill_category": str(skill.get("category", "") or ""),
        "rank": rank,
        "preview": content[:160],
        "text_for_overlap": text,
    }


def serialize_memory(memory_unit, rank: int) -> dict[str, Any]:
    text = " ".join(
        part
        for part in [
            getattr(memory_unit, "summary", "") or "",
            getattr(memory_unit, "content", "") or "",
        ]
        if part
    ).strip()
    return {
        "kind": "memory",
        "memory_id": getattr(memory_unit, "memory_id", ""),
        "memory_type": getattr(getattr(memory_unit, "memory_type", None), "value", ""),
        "rank": rank,
        "importance": round(float(getattr(memory_unit, "importance", 0.0) or 0.0), 4),
        "preview": str(getattr(memory_unit, "content", "") or "")[:160],
        "text_for_overlap": text,
    }


def _score_items(response_text: str, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    scored: list[dict[str, Any]] = []
    bundle_text_parts: list[str] = []
    for item in items:
        item_text = str(item.get("text_for_overlap", "") or "")
        bundle_text_parts.append(item_text)
        overlap = overlap_coefficient(response_text, item_text)
        cleaned = {key: value for key, value in item.items() if key != "text_for_overlap"}
        cleaned["response_overlap"] = overlap
        scored.append(cleaned)

    bundle_overlap = overlap_coefficient(response_text, "\n".join(bundle_text_parts))
    scored.sort(key=lambda entry: entry.get("response_overlap", 0.0), reverse=True)
    return scored, bundle_overlap


def finalize_provenance(
    response_text: str,
    *,
    mode: str,
    skill_generation: int,
    memory_scope: str,
    memory_policy_version: int | None,
    skills: list[dict[str, Any]] | None = None,
    memories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    skills = skills or []
    memories = memories or []
    scored_skills, skill_bundle_overlap = _score_items(response_text, skills)
    scored_memories, memory_bundle_overlap = _score_items(response_text, memories)
    non_policy_proxy = min(1.0, skill_bundle_overlap + memory_bundle_overlap)
    return {
        "mode": mode,
        "skill_generation": skill_generation,
        "memory_scope": memory_scope,
        "memory_policy_version": memory_policy_version,
        "skill_count": len(scored_skills),
        "memory_count": len(scored_memories),
        "skill_bundle_overlap": skill_bundle_overlap,
        "memory_bundle_overlap": memory_bundle_overlap,
        "policy_residual_proxy": round(max(0.0, 1.0 - non_policy_proxy), 4),
        "skills": scored_skills,
        "memories": scored_memories,
    }

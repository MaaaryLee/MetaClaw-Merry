from __future__ import annotations

from collections import Counter


def _scores(record: dict) -> tuple[float, float, float]:
    skill = float(record.get("skill_contribution_score", 0.0) or 0.0)
    memory = float(record.get("memory_contribution_score", 0.0) or 0.0)
    policy = float(record.get("policy_residual_proxy", 0.0) or 0.0)
    return skill, memory, policy


def is_effective(record: dict) -> bool:
    return int(record.get("loss_mask_sum", 0) or 0) > 0


def route_naive(record: dict) -> str:
    return "policy" if is_effective(record) else "excluded"


def route_heuristic(
    record: dict,
    *,
    policy_threshold: float = 0.45,
    skill_threshold: float = 0.55,
    memory_threshold: float = 0.55,
) -> str:
    if not is_effective(record):
        return "excluded"
    skill, memory, policy = _scores(record)
    if skill >= skill_threshold and skill > memory and skill > policy:
        return "skill"
    if memory >= memory_threshold and memory > skill and memory > policy:
        return "memory"
    if policy >= policy_threshold:
        return "policy"
    return "mixed"


def route_attribution_aware(
    record: dict,
    *,
    policy_threshold: float = 0.35,
    ambiguity_margin: float = 0.08,
) -> str:
    if not is_effective(record):
        return "excluded"
    skill, memory, policy = _scores(record)
    ranked = sorted(
        [("skill", skill), ("memory", memory), ("policy", policy)],
        key=lambda item: item[1],
        reverse=True,
    )
    top_name, top_value = ranked[0]
    second_value = ranked[1][1]
    if top_name == "policy" and top_value < policy_threshold:
        return "mixed"
    if (top_value - second_value) <= ambiguity_margin:
        return "mixed"
    return top_name


def compare_routing_strategies(records: list[dict]) -> dict:
    strategies = {
        "naive": route_naive,
        "heuristic": route_heuristic,
        "attribution_aware": route_attribution_aware,
    }
    comparison: dict[str, dict] = {}
    for name, router in strategies.items():
        labels = [router(record) for record in records]
        counts = Counter(labels)
        comparison[name] = {
            "route_counts": dict(counts),
            "policy_updates": counts.get("policy", 0),
        }
    naive_updates = comparison["naive"]["policy_updates"]
    for name, payload in comparison.items():
        payload["policy_update_reduction"] = naive_updates - payload["policy_updates"]
    return comparison


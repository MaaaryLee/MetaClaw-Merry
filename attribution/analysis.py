from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .router import compare_routing_strategies, route_attribution_aware
except ImportError:
    from router import compare_routing_strategies, route_attribution_aware


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    return records


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / float(len(values)), 4)


def _scenario_of(record: dict) -> str:
    scenario = str(record.get("scenario", "") or "")
    if scenario:
        return scenario
    provenance = record.get("provenance", {}) or {}
    return str(provenance.get("mode", "unknown") or "unknown")


def compute_grpo_advantages(
    records: list[dict],
    *,
    use_attribution: bool = False,
    policy_residual_floor: float = 0.0,
) -> list[float]:
    if not records:
        return []
    floor = max(0.0, min(1.0, policy_residual_floor))
    rewards: list[float] = []
    for record in records:
        reward = float(record.get("reward", 0.0) or 0.0)
        if use_attribution:
            policy_proxy = float(record.get("policy_residual_proxy", 0.0) or 0.0)
            reward *= max(floor, min(1.0, policy_proxy))
        rewards.append(reward)
    mean_r = sum(rewards) / len(rewards)
    variance = sum((reward - mean_r) ** 2 for reward in rewards) / len(rewards)
    std_r = variance ** 0.5
    eps = 1e-8
    return [round((reward - mean_r) / (std_r + eps), 4) for reward in rewards]


def build_summary(records: list[dict]) -> dict:
    if not records:
        return {"sample_count": 0, "message": "No records found."}

    effective = [record for record in records if int(record.get("loss_mask_sum", 0) or 0) > 0]
    rewards = [float(record.get("reward", 0.0) or 0.0) for record in records]
    skill_scores = [float(record.get("skill_contribution_score", 0.0) or 0.0) for record in records]
    memory_scores = [float(record.get("memory_contribution_score", 0.0) or 0.0) for record in records]
    policy_scores = [float(record.get("policy_residual_proxy", 0.0) or 0.0) for record in records]
    routing = compare_routing_strategies(records)

    scenario_breakdown: dict[str, dict[str, int]] = {}
    for record in records:
        scenario = _scenario_of(record)
        label = route_attribution_aware(record)
        bucket = scenario_breakdown.setdefault(scenario, {})
        bucket[label] = bucket.get(label, 0) + 1

    return {
        "sample_count": len(records),
        "effective_sample_count": len(effective),
        "mean_reward": _mean(rewards),
        "mean_skill_contribution": _mean(skill_scores),
        "mean_memory_contribution": _mean(memory_scores),
        "mean_policy_residual_proxy": _mean(policy_scores),
        "routing": routing,
        "scenario_breakdown": scenario_breakdown,
        "baseline_advantages": compute_grpo_advantages(effective),
        "attribution_weighted_advantages": compute_grpo_advantages(
            effective,
            use_attribution=True,
            policy_residual_floor=0.1,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize attribution-aware MetaClaw sample logs.")
    parser.add_argument(
        "--records",
        default="attribution/demo_training_samples.jsonl",
        help="Path to a training_samples-style JSONL file.",
    )
    parser.add_argument(
        "--summary-out",
        default="",
        help="Optional path to write the JSON summary.",
    )
    args = parser.parse_args()

    records = load_records(Path(args.records).expanduser())
    summary = build_summary(records)
    if args.summary_out:
        summary_path = Path(args.summary_out).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SCENARIO_COUNTS = {
    "policy_success": 18,
    "skill_success": 16,
    "memory_success": 14,
    "mixed_success": 16,
    "ambiguous_success": 10,
    "policy_failure": 10,
    "augmentation_failure": 8,
}


def _make_record(scenario: str, idx: int, rng: random.Random) -> dict:
    if scenario == "policy_success":
        skill, memory, policy, reward, mode = rng.uniform(0.05, 0.18), rng.uniform(0.03, 0.16), rng.uniform(0.72, 0.92), 1.0, "none"
    elif scenario == "skill_success":
        skill, memory, policy, reward, mode = rng.uniform(0.62, 0.85), rng.uniform(0.02, 0.18), rng.uniform(0.12, 0.32), 1.0, "skills"
    elif scenario == "memory_success":
        skill, memory, policy, reward, mode = rng.uniform(0.04, 0.2), rng.uniform(0.6, 0.86), rng.uniform(0.1, 0.3), 1.0, "memory"
    elif scenario == "mixed_success":
        skill, memory, policy, reward, mode = rng.uniform(0.35, 0.55), rng.uniform(0.28, 0.48), rng.uniform(0.18, 0.34), 1.0, "synergy"
    elif scenario == "ambiguous_success":
        skill, memory, policy, reward, mode = rng.uniform(0.28, 0.4), rng.uniform(0.26, 0.4), rng.uniform(0.28, 0.4), 1.0, "synergy"
    elif scenario == "policy_failure":
        skill, memory, policy, reward, mode = rng.uniform(0.04, 0.16), rng.uniform(0.02, 0.14), rng.uniform(0.7, 0.9), -1.0, "none"
    else:
        skill, memory, policy, reward, mode = rng.uniform(0.34, 0.56), rng.uniform(0.3, 0.52), rng.uniform(0.05, 0.24), -1.0, "synergy"

    record = {
        "session_id": f"demo-session-{idx // 3 + 1:03d}",
        "turn_num": idx + 1,
        "reward": reward,
        "loss_mask_sum": rng.randint(3, 9),
        "prompt_token_count": rng.randint(120, 480),
        "response_token_count": rng.randint(40, 220),
        "skill_generation": rng.randint(1, 4),
        "skill_names": [f"skill-{(idx % 5) + 1}"] if skill > 0.2 else [],
        "memory_ids": [f"mem-{(idx % 7) + 1}"] if memory > 0.2 else [],
        "skill_contribution_score": round(skill, 4),
        "memory_contribution_score": round(memory, 4),
        "policy_residual_proxy": round(policy, 4),
        "prm_votes": [1, 1, 1] if reward > 0 else [-1, -1, -1],
        "scenario": scenario,
        "task_family": rng.choice(["debugging", "project-memory", "workflow", "planning"]),
        "provenance": {
            "mode": mode,
            "skill_bundle_overlap": round(skill, 4),
            "memory_bundle_overlap": round(memory, 4),
            "policy_residual_proxy": round(policy, 4),
            "skills": [{"skill_name": f"skill-{(idx % 5) + 1}"}] if skill > 0.2 else [],
            "memories": [{"memory_id": f"mem-{(idx % 7) + 1}"}] if memory > 0.2 else [],
        },
    }
    return record


def generate_records(seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []
    idx = 0
    for scenario, count in SCENARIO_COUNTS.items():
        for _ in range(count):
            records.append(_make_record(scenario, idx, rng))
            idx += 1
    rng.shuffle(records)
    for i, record in enumerate(records, start=1):
        record["turn_num"] = i
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic attribution demo records.")
    parser.add_argument(
        "--out",
        default="attribution/demo_training_samples.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = generate_records(seed=args.seed)
    out_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"sample_count": len(records), "out": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from statistics import pstdev
import sys

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency in some environments
    yaml = None

try:
    from .analysis import load_records
except ImportError:
    from analysis import load_records

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _extract_instruction_text(prompt_text: str) -> str:
    lines = [line.strip() for line in prompt_text.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.lower().startswith("user:"):
            return line.split(":", 1)[1].strip()
    return prompt_text.strip()


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / float(len(values)), 4)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return round(float(pstdev(values)), 4)


def _normalize_rewards(rewards: list[float]) -> list[float]:
    if not rewards:
        return []
    mean_r = sum(rewards) / len(rewards)
    variance = sum((reward - mean_r) ** 2 for reward in rewards) / len(rewards)
    std_r = variance ** 0.5
    eps = 1e-8
    return [round((reward - mean_r) / (std_r + eps), 4) for reward in rewards]


def _top_skill_name(record: dict) -> str:
    provenance = record.get("provenance", {}) or {}
    skills = provenance.get("skills", []) or []
    if not skills:
        return ""
    return str(skills[0].get("skill_name", "") or "")


def _load_llm_settings(config_path: Path) -> dict[str, str]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read the config file. Try `pip install pyyaml`.")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    llm = payload.get("llm", {}) or {}
    return {
        "api_base": str(llm.get("api_base", "") or ""),
        "api_key": str(llm.get("api_key", "") or ""),
        "model_id": str(llm.get("model_id", "") or ""),
    }


async def _score_records(records: list[dict], scorer: PRMScorer) -> list[dict]:
    scored: list[dict] = []
    for index, record in enumerate(records, start=1):
        prompt_text = str(record.get("prompt_text", "") or "")
        response_text = str(record.get("response_text", "") or "")
        instruction_text = _extract_instruction_text(prompt_text)
        if not instruction_text or not response_text:
            continue
        result = await scorer.evaluate(
            response=response_text,
            instruction=instruction_text,
            session_id=str(record.get("session_id", "") or ""),
            turn_num=int(record.get("turn_num", 0) or 0),
        )
        scored_record = dict(record)
        scored_record["ab_index"] = index
        scored_record["instruction_text"] = instruction_text
        scored_record["reward"] = float(result.get("score", 0.0) or 0.0)
        scored_record["prm_votes"] = list(result.get("votes", []))
        scored_record["eval_text"] = str(result.get("eval_text", "") or "")
        scored.append(scored_record)
    return scored


def _build_ab_summary(records: list[dict]) -> dict:
    rewards = [float(record.get("reward", 0.0) or 0.0) for record in records]
    weighted_rewards = [
        round(
            float(record.get("reward", 0.0) or 0.0)
            * float(record.get("policy_residual_proxy", 0.0) or 0.0),
            4,
        )
        for record in records
    ]
    baseline_advantages = _normalize_rewards(rewards)
    weighted_advantages = _normalize_rewards(weighted_rewards)

    return {
        "sample_count": len(records),
        "reward_distribution": {
            "-1": sum(1 for reward in rewards if reward < 0),
            "0": sum(1 for reward in rewards if reward == 0),
            "1": sum(1 for reward in rewards if reward > 0),
        },
        "mean_skill_contribution": _mean(
            [float(record.get("skill_contribution_score", 0.0) or 0.0) for record in records]
        ),
        "mean_policy_residual_proxy": _mean(
            [float(record.get("policy_residual_proxy", 0.0) or 0.0) for record in records]
        ),
        "baseline_reward_mean": _mean(rewards),
        "baseline_reward_std": _std(rewards),
        "weighted_reward_mean": _mean(weighted_rewards),
        "weighted_reward_std": _std(weighted_rewards),
        "baseline_advantages": baseline_advantages,
        "weighted_advantages": weighted_advantages,
        "baseline_zero_advantages": sum(1 for value in baseline_advantages if abs(value) < 1e-9),
        "weighted_zero_advantages": sum(1 for value in weighted_advantages if abs(value) < 1e-9),
        "records": [
            {
                "session_id": record.get("session_id", ""),
                "reward": record.get("reward", 0.0),
                "policy_residual_proxy": record.get("policy_residual_proxy", 0.0),
                "weighted_reward": round(
                    float(record.get("reward", 0.0) or 0.0)
                    * float(record.get("policy_residual_proxy", 0.0) or 0.0),
                    4,
                ),
                "skill_contribution_score": record.get("skill_contribution_score", 0.0),
                "top_skill": _top_skill_name(record),
                "prm_votes": record.get("prm_votes", []),
            }
            for record in records
        ],
    }


async def _async_main(args) -> None:
    import importlib.util

    prm_scorer_path = REPO_ROOT / "metaclaw" / "prm_scorer.py"
    prm_spec = importlib.util.spec_from_file_location("metaclaw_prm_scorer", prm_scorer_path)
    if prm_spec is None or prm_spec.loader is None:
        raise RuntimeError(f"Could not load PRM scorer module from {prm_scorer_path}")
    prm_module = importlib.util.module_from_spec(prm_spec)
    prm_spec.loader.exec_module(prm_module)
    PRMScorer = prm_module.PRMScorer

    config = _load_llm_settings(Path(args.config).expanduser())
    scorer = PRMScorer(
        prm_url=args.judge_base_url or config["api_base"],
        prm_model=args.judge_model or config["model_id"],
        api_key=args.judge_api_key or config["api_key"],
        prm_m=args.prm_votes,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    records = load_records(Path(args.records).expanduser())
    scored_records = await _score_records(records, scorer)
    summary = _build_ab_summary(scored_records)

    if args.scored_out:
        scored_out = Path(args.scored_out).expanduser()
        scored_out.parent.mkdir(parents=True, exist_ok=True)
        scored_out.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in scored_records) + ("\n" if scored_records else ""),
            encoding="utf-8",
        )
    if args.summary_out:
        summary_out = Path(args.summary_out).expanduser()
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline A/B for MetaClaw training signals: raw reward vs attribution-weighted reward."
    )
    parser.add_argument(
        "--records",
        default="records/training_samples.jsonl",
        help="Path to attribution-aware training_samples JSONL.",
    )
    parser.add_argument(
        "--config",
        default="benchmark/scripts/config/skills-only.local.yaml",
        help="Local YAML config used to populate the judge API base/key/model.",
    )
    parser.add_argument("--judge-base-url", default="", help="Override judge API base URL.")
    parser.add_argument("--judge-model", default="", help="Override judge model name.")
    parser.add_argument("--judge-api-key", default="", help="Override judge API key.")
    parser.add_argument("--prm-votes", type=int, default=3, help="Majority-vote count for PRM scoring.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Judge sampling temperature.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max judge output tokens.")
    parser.add_argument(
        "--scored-out",
        default="records/training_samples_scored.jsonl",
        help="Where to save the scored records JSONL.",
    )
    parser.add_argument(
        "--summary-out",
        default="records/training_signal_ab_summary.json",
        help="Where to save the A/B summary JSON.",
    )
    args = parser.parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()

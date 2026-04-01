from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

try:
    from .analysis import build_summary
    from .generate_synthetic_data import generate_records
except ImportError:
    from analysis import build_summary
    from generate_synthetic_data import generate_records


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "attribution" / "local_check_output.json"


def _load_module(module_name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    attribution = _load_module("metaclaw_attribution_local", "metaclaw/attribution.py")
    formatter = _load_module("metaclaw_data_formatter_local", "metaclaw/data_formatter.py")

    skill = attribution.serialize_skill(
        {
            "name": "debug-systematically",
            "description": "Use logs before editing code",
            "content": "check logs isolate issue reproduce bug before editing code",
        },
        rank=1,
    )
    memory = attribution.serialize_memory(
        type(
            "MemoryUnit",
            (),
            {
                "memory_id": "mem-1",
                "summary": "Project prefers small patches and local checks",
                "content": "keep changes minimal and test locally before large refactors",
                "memory_type": type("MemoryType", (), {"value": "preference"})(),
                "importance": 0.8,
            },
        )(),
        rank=1,
    )
    provenance = attribution.finalize_provenance(
        "check logs first and keep the patch small before editing code",
        mode="synergy",
        skill_generation=2,
        memory_scope="default",
        memory_policy_version=1,
        skills=[skill],
        memories=[memory],
    )

    Sample = formatter.ConversationSample
    batch = [
        Sample("demo", 1, [1, 2], [3, 4], [-0.2, -0.1], [1, 1], 1.0, policy_residual_proxy=0.2),
        Sample("demo", 2, [1, 2], [5, 6], [-0.3, -0.2], [1, 1], 1.0, policy_residual_proxy=0.9),
        Sample("demo", 3, [1, 2], [7, 8], [-0.4, -0.3], [1, 1], -1.0, policy_residual_proxy=0.8),
    ]
    base_adv = formatter.compute_advantages(batch)
    weighted_adv = formatter.compute_advantages(
        batch,
        use_attribution=True,
        policy_residual_floor=0.1,
    )

    records = generate_records(seed=11)[:18]
    summary = build_summary(records)
    report = {
        "phase_1_provenance": {
            "skill_bundle_overlap": provenance["skill_bundle_overlap"],
            "memory_bundle_overlap": provenance["memory_bundle_overlap"],
            "policy_residual_proxy": provenance["policy_residual_proxy"],
            "skill_count": provenance["skill_count"],
            "memory_count": provenance["memory_count"],
        },
        "phase_2_offline_analysis": {
            "sample_count": summary["sample_count"],
            "routing": summary["routing"],
        },
        "phase_3_advantages": {
            "baseline": [round(value, 4) for value in base_adv],
            "attribution_weighted": [round(value, 4) for value in weighted_adv],
        },
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


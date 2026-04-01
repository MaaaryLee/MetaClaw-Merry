from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="metaclaw-mpl-"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .analysis import build_summary, load_records
except ImportError:
    from analysis import build_summary, load_records


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_figures(records: list[dict], summary: dict, out_dir: Path) -> list[str]:
    out_dir = _ensure_dir(out_dir)
    figure_paths: list[str] = []

    skill = np.array([float(record.get("skill_contribution_score", 0.0) or 0.0) for record in records])
    memory = np.array([float(record.get("memory_contribution_score", 0.0) or 0.0) for record in records])
    policy = np.array([float(record.get("policy_residual_proxy", 0.0) or 0.0) for record in records])

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 15)
    ax.hist(skill, bins=bins, alpha=0.55, label="skill", color="#d95f02")
    ax.hist(memory, bins=bins, alpha=0.55, label="memory", color="#1b9e77")
    ax.hist(policy, bins=bins, alpha=0.55, label="policy", color="#7570b3")
    ax.set_title("Attribution Score Distributions")
    ax.set_xlabel("Contribution proxy")
    ax.set_ylabel("Sample count")
    ax.legend(frameon=False)
    path1 = out_dir / "01_score_distributions.png"
    fig.tight_layout()
    fig.savefig(path1, dpi=180)
    plt.close(fig)
    figure_paths.append(str(path1))

    routing = summary.get("routing", {})
    strategies = ["naive", "heuristic", "attribution_aware"]
    policy_updates = [routing.get(name, {}).get("policy_updates", 0) for name in strategies]
    reductions = [routing.get(name, {}).get("policy_update_reduction", 0) for name in strategies]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(strategies))
    ax.bar(x, policy_updates, color=["#9ecae1", "#6baed6", "#2171b5"])
    for i, reduction in enumerate(reductions):
        ax.text(i, policy_updates[i] + 0.5, f"-{reduction}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["naive", "heuristic", "attr-aware"])
    ax.set_ylabel("Samples routed to policy")
    ax.set_title("Routing Comparison Across Strategies")
    path2 = out_dir / "02_routing_comparison.png"
    fig.tight_layout()
    fig.savefig(path2, dpi=180)
    plt.close(fig)
    figure_paths.append(str(path2))

    baseline = np.array(summary.get("baseline_advantages", []))
    weighted = np.array(summary.get("attribution_weighted_advantages", []))
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(min(baseline.min(initial=-2), weighted.min(initial=-2)), max(baseline.max(initial=2), weighted.max(initial=2)), 14)
    ax.hist(baseline, bins=bins, alpha=0.6, label="baseline", color="#636363")
    ax.hist(weighted, bins=bins, alpha=0.6, label="attribution-weighted", color="#3182bd")
    ax.set_title("GRPO Advantage Distribution Before vs After Attribution Weighting")
    ax.set_xlabel("Advantage")
    ax.set_ylabel("Effective sample count")
    ax.legend(frameon=False)
    path3 = out_dir / "03_advantage_shift.png"
    fig.tight_layout()
    fig.savefig(path3, dpi=180)
    plt.close(fig)
    figure_paths.append(str(path3))

    scenarios = summary.get("scenario_breakdown", {})
    scenario_names = list(scenarios.keys())
    policy_counts = [scenarios[name].get("policy", 0) for name in scenario_names]
    skill_counts = [scenarios[name].get("skill", 0) for name in scenario_names]
    memory_counts = [scenarios[name].get("memory", 0) for name in scenario_names]
    mixed_counts = [scenarios[name].get("mixed", 0) for name in scenario_names]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(scenario_names, policy_counts, label="policy", color="#7570b3")
    ax.bar(scenario_names, skill_counts, bottom=policy_counts, label="skill", color="#d95f02")
    ax.bar(
        scenario_names,
        memory_counts,
        bottom=np.array(policy_counts) + np.array(skill_counts),
        label="memory",
        color="#1b9e77",
    )
    ax.bar(
        scenario_names,
        mixed_counts,
        bottom=np.array(policy_counts) + np.array(skill_counts) + np.array(memory_counts),
        label="mixed",
        color="#bdbdbd",
    )
    ax.set_title("Scenario Breakdown of Attribution-Aware Routing")
    ax.set_ylabel("Sample count")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False, ncol=4)
    path4 = out_dir / "04_scenario_breakdown.png"
    fig.tight_layout()
    fig.savefig(path4, dpi=180)
    plt.close(fig)
    figure_paths.append(str(path4))

    return figure_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate attribution demo figures.")
    parser.add_argument("--records", default="attribution/demo_training_samples.jsonl")
    parser.add_argument("--out-dir", default="attribution/figures")
    args = parser.parse_args()

    records = load_records(Path(args.records).expanduser())
    summary = build_summary(records)
    paths = make_figures(records, summary, Path(args.out_dir).expanduser())
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()


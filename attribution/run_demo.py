from __future__ import annotations

import json
from pathlib import Path

try:
    from .analysis import build_summary, load_records
    from .generate_synthetic_data import generate_records
    from .visualize import make_figures
except ImportError:
    from analysis import build_summary, load_records
    from generate_synthetic_data import generate_records
    from visualize import make_figures


ROOT = Path(__file__).resolve().parent
DEMO_RECORDS = ROOT / "demo_training_samples.jsonl"
DEMO_SUMMARY = ROOT / "demo_summary.json"
FIGURE_DIR = ROOT / "figures"


def main() -> None:
    records = generate_records(seed=7)
    DEMO_RECORDS.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    loaded = load_records(DEMO_RECORDS)
    summary = build_summary(loaded)
    figure_paths = make_figures(loaded, summary, FIGURE_DIR)
    summary["generated_figures"] = [str(Path(path).relative_to(ROOT)) for path in figure_paths]
    DEMO_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


# MetaClaw Attribution Extension

## Abstract

MetaClaw combines three adaptation channels at inference time: the base policy, retrieved skills, and long-term memory. A successful response may improve because the model itself adapted, because a skill provided the right procedure, or because memory supplied missing project context. This extension adds a lightweight attribution layer to MetaClaw so those cases are no longer treated as one undifferentiated training signal. It instruments the serving path to log augmentation provenance, computes overlap-based contribution proxies for skills, memory, and residual policy behavior, and supports offline routing analysis over where future learning should go. The goal is not to claim causal attribution, but to make continual learning in MetaClaw more attributable before changing the underlying learning rule.

## What this extension is for

This advisor-demo prototype tries to answer one question:

**When a MetaClaw response succeeds, should that signal update the base policy, become a reusable skill, or stay as memory?**

It does that in three phases:

1. `metaclaw/` patch:
   record provenance in the real serving pipeline
2. `attribution/` analysis:
   inspect sample logs and compare routing strategies
3. optional training hook:
   allow attribution-weighted advantages, but keep it off by default

## Directory guide

- `analysis.py`
  Loads `training_samples`-style JSONL records and produces attribution summaries.

- `router.py`
  Compares three routing strategies: naive, heuristic, and attribution-aware.

- `visualize.py`
  Generates the four advisor-facing figures.

- `generate_synthetic_data.py`
  Creates reproducible demo records with seven scenario types. This is support infrastructure only, not the main empirical claim.

- `run_demo.py`
  One command to generate demo records, summary JSON, and figures.

- `run_local_checks.py`
  Small smoke test for all three phases.

## Important scope note

The primary target of this extension is **real provenance-aware logs from the MetaClaw pipeline**.

The synthetic generator in this directory exists only to:
- make the demo reproducible
- let us bootstrap analysis and figures without the full RL stack
- provide a clean walkthrough artifact for an advisor

It should **not** be presented as proof that attribution is causal or validated.

## Commands to run locally

### 1. Generate the full demo package

```bash
MPLCONFIGDIR=/tmp/metaclaw-mpl python3 attribution/run_demo.py
```

Success looks like:
- `attribution/demo_training_samples.jsonl` exists
- `attribution/demo_summary.json` exists
- `attribution/figures/` contains 4 PNG files

Failure looks like:
- Python import errors
- or matplotlib cache permission issues if `MPLCONFIGDIR` is not writable

### 2. Run the local smoke checks

```bash
python3 attribution/run_local_checks.py
```

Success looks like:
- JSON with `phase_1_provenance`
- JSON with `phase_2_offline_analysis`
- JSON with `phase_3_advantages`

### 3. Analyze a real sample log later

```bash
python3 attribution/analysis.py \
  --records records/training_samples.jsonl \
  --summary-out records/attribution_summary.json
```

Success looks like:
- non-zero sample counts
- routing counts under `naive`, `heuristic`, and `attribution_aware`

## Figure outputs

After `run_demo.py`, the advisor-facing figures live in `attribution/figures/`:

1. score distributions
2. routing comparison
3. GRPO advantage shift
4. scenario breakdown

### Preview

![Score distributions](figures/01_score_distributions.png)
![Routing comparison](figures/02_routing_comparison.png)
![Advantage shift](figures/03_advantage_shift.png)
![Scenario breakdown](figures/04_scenario_breakdown.png)

## Environment note

The offline demo and visualization stack do **not** require the full MetaClaw RL runtime.

If your local environment is missing serving dependencies such as `uvicorn`, you can still run:
- `attribution/run_demo.py`
- `attribution/run_local_checks.py`
- `attribution/analysis.py`

That separation is intentional for advisor-facing reproducibility.

## Design discipline

- backward compatible by default
- attribution-weighted training is optional and experimental
- no claim that the overlap proxy is causal
- no requirement to boot the full MetaClaw training stack just to view the demo

# Adaptation Under Perturbations in Self-Play RL

A minimal threshold in decision capacity determines whether self-play RL
agents collapse under asymmetric rule changes. Tested in Kuhn and Leduc
Poker with CFR, Q-learning, and a frozen Q-learning baseline.

**Paper:** [report/paper.md](report/paper.md)

## Key Results

- **Removing all decisions** from a player causes Q-learning to collapse
  to near-maximal exploitation (Kuhn: -0.93, Leduc: -0.31). The collapse
  is co-adaptation-driven -- a frozen Q-learning baseline avoids it.
- **Preserving a single decision** stabilises Q-learning near Nash
  equilibrium (Kuhn: -0.07, Leduc: -0.10). The threshold is sharp, not
  gradual.

## Quick Start

```bash
pip install -r requirements.txt
python run_experiments.py
```

## All Experiments

| Script | What it runs |
|---|---|
| `python run_experiments.py` | Base Kuhn experiments (full removal + root-only) |
| `python run_leduc_experiments.py` | Leduc Poker experiments |
| `python run_capacity_sweep.py` | Decision capacity sweep (0, 1, 2) |
| `python run_severity_sweep.py` | Perturbation timing x severity (3x2 grid) |
| `python run_cross_game.py` | Kuhn vs Leduc side-by-side comparison |
| `python run_variance_decomposition.py` | Environment vs policy variance analysis |

Single experiment:

```bash
python run_experiments.py --config configs/frozen/frozen_full_removal.yaml
```

## Project Structure

```
configs/                  Experiment configs (YAML)
  full_removal.yaml       Base Kuhn experiments
  root_only.yaml
  capacity/               Decision capacity sweep
  frozen/                 Frozen Q-learning comparison
  leduc/                  Leduc Poker experiments
  severity/               Timing x severity sweep
  stochastic/             Stochastic masking
src/
  env/                    Game environments + perturbation wrappers
  agents/                 CFR, Q-learning, frozen Q-learning
  experiments/runner.py   Multi-seed experiment runner
  utils/                  Metrics, plotting, variance decomposition
results/
  plots/                  Generated figures (tracked)
  raw/                    Per-seed CSVs (regenerated, gitignored)
report/paper.md           Full writeup with results and references
```

## Requirements

Python 3.10+. Dependencies: `numpy`, `matplotlib`, `pyyaml`, `scipy`.
No external game libraries -- both poker variants are implemented from scratch.

"""Run decision capacity sweep: experiments at capacity 0, 1, and 2.

Usage:
    python run_capacity_sweep.py
"""

from pathlib import Path

from src.config_loader import load_config
from src.experiments.runner import run_experiment
from src.utils.plotting import plot_capacity_sweep


CAPACITY_CONFIGS = [
    ("configs/capacity/capacity_0.yaml", 0),
    ("configs/capacity/capacity_1.yaml", 1),
    ("configs/capacity/capacity_2.yaml", 2),
]


def main():
    sweep_results = {}

    for path, capacity in CAPACITY_CONFIGS:
        print(f"\n>>> Capacity {capacity}: {path}")
        config = load_config(path)
        _, _, stat_summary = run_experiment(config)
        sweep_results[capacity] = stat_summary

    plot_path = Path("results/plots") / "capacity_sweep.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_capacity_sweep(sweep_results, str(plot_path))

    print(f"\n{'=' * 60}")
    print(f"  Capacity sweep complete.  Plot -> {plot_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

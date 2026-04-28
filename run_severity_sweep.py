"""Run perturbation timing x severity sweep.

Usage:
    python run_severity_sweep.py
"""

from pathlib import Path

from src.config_loader import load_config
from src.experiments.runner import run_experiment
from src.utils.plotting import plot_severity_sweep


SWEEP_CONFIGS = [
    ("configs/severity/early_severe.yaml", "early", "severe"),
    ("configs/severity/early_mild.yaml", "early", "mild"),
    ("configs/severity/mid_severe.yaml", "mid", "severe"),
    ("configs/severity/mid_mild.yaml", "mid", "mild"),
    ("configs/severity/late_severe.yaml", "late", "severe"),
    ("configs/severity/late_mild.yaml", "late", "mild"),
]


def main():
    sweep_results = {}

    for path, timing, severity in SWEEP_CONFIGS:
        print(f"\n>>> {timing} x {severity}: {path}")
        config = load_config(path)
        _, _, stat_summary = run_experiment(config)
        sweep_results[(timing, severity)] = stat_summary

    plot_path = Path("results/plots") / "severity_sweep.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_severity_sweep(sweep_results, str(plot_path))

    print(f"\n{'=' * 60}")
    print(f"  Severity sweep complete.  Plot -> {plot_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

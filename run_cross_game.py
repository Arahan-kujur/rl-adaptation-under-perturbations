"""Run cross-game comparison: same perturbation in Kuhn and Leduc.

Usage:
    python run_cross_game.py
"""

from pathlib import Path

from src.config_loader import load_config
from src.experiments.runner import run_experiment
from src.utils.plotting import plot_cross_game
from src.utils.metrics import (
    collapse_summary, format_collapse_table,
)


CONFIGS = {
    "kuhn": "configs/full_removal.yaml",
    "leduc": "configs/leduc/leduc_full_removal.yaml",
}


def main():
    game_stats = {}

    for game, path in CONFIGS.items():
        print(f"\n>>> {game}: {path}")
        config = load_config(path)
        _, _, stat_summary = run_experiment(config)
        game_stats[game] = stat_summary

    plot_path = Path("results/plots") / "cross_game.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_cross_game(game_stats, str(plot_path))

    print(f"\n{'=' * 60}")
    print(f"  Cross-game comparison complete.  Plot -> {plot_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

"""Run variance decomposition analysis on one or more experiment configs.

Usage:
    python run_variance_decomposition.py                                  # both base configs
    python run_variance_decomposition.py --config configs/root_only.yaml  # single config
"""

import argparse
from pathlib import Path

from src.config_loader import load_config
from src.utils.variance_decomposition import run_decomposition


def main():
    parser = argparse.ArgumentParser(
        description="Variance decomposition: environment vs policy randomness.")
    parser.add_argument(
        "--config", type=Path, nargs="+",
        help="Config file(s). Defaults to both base configs.")
    args = parser.parse_args()

    if args.config:
        config_paths = args.config
    else:
        config_paths = sorted(Path("configs").glob("*.yaml"))

    if not config_paths:
        print("No config files found.")
        raise SystemExit(1)

    for path in config_paths:
        print(f"\nLoading config: {path}")
        config = load_config(path)
        run_decomposition(config)

    print(f"\n{'=' * 60}")
    print("  Variance decomposition complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

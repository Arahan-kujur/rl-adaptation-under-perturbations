"""Variance decomposition: environment randomness vs policy randomness."""

import numpy as np
from pathlib import Path

from src.env.game_registry import get_game
from src.experiments.runner import run_single_seed
from src.utils.plotting import plot_variance_decomposition


def run_decomposition(config, reference_seed=42):
    """Run variance decomposition for an experiment config.

    Executes three conditions across the configured seeds:
    1. Total -- both environment and policy RNGs vary per seed.
    2. Policy only -- environment RNG fixed, policy varies.
    3. Environment only -- policy RNG fixed, environment varies.
    """
    name = config["experiment"]["name"]
    seeds = config["experiment"]["seeds"]
    cfr_iters = config["cfr"]["iterations"]
    game_name = config["experiment"].get("game", "kuhn")
    include_frozen = "q_learning_frozen" in config

    game_info = get_game(game_name)

    print(f"\n{'=' * 60}")
    print(f"  Variance Decomposition: {name}  "
          f"({len(seeds)} seeds, game={game_name})")
    print(f"{'=' * 60}")

    trainer = game_info["cfr_trainer_class"]()
    trainer.train(cfr_iters)
    policy = trainer.get_average_strategy()

    # ---- Condition 1: total variance ----
    print("\n  [1/3] Total variance (both RNG sources vary)...")
    total_summaries = []
    for seed in seeds:
        _, _, summary = run_single_seed(
            seed, policy, config, game_info,
            include_frozen=include_frozen,
            csv_suffix="_var_total", quiet=True)
        total_summaries.append(summary)

    # ---- Condition 2: policy-only variance ----
    print("  [2/3] Policy variance (env fixed, policy varies)...")
    policy_summaries = []
    for seed in seeds:
        _, _, summary = run_single_seed(
            seed, policy, config, game_info,
            include_frozen=include_frozen,
            override_env_seed=reference_seed,
            csv_suffix="_var_policy", quiet=True)
        policy_summaries.append(summary)

    # ---- Condition 3: env-only variance ----
    print("  [3/3] Environment variance (policy fixed, env varies)...")
    env_summaries = []
    for seed in seeds:
        _, _, summary = run_single_seed(
            seed, policy, config, game_info,
            include_frozen=include_frozen,
            override_policy_seed=reference_seed,
            csv_suffix="_var_env", quiet=True)
        env_summaries.append(summary)

    # ---- Compute variance table ----
    agents = [k for k in total_summaries[0] if k != "_meta"]
    var_table = {}

    print(f"\n  {'Agent':14s}  {'V_total':>10s}  {'V_env':>10s}  "
          f"{'V_policy':>10s}  {'V_interact':>10s}")
    print("  " + "-" * 62)

    for agent in agents:
        v_total = float(np.var(
            [s[agent]["post"] for s in total_summaries]))
        v_policy = float(np.var(
            [s[agent]["post"] for s in policy_summaries]))
        v_env = float(np.var(
            [s[agent]["post"] for s in env_summaries]))
        v_interaction = v_total - v_env - v_policy

        var_table[agent] = {
            "total": v_total, "env": v_env,
            "policy": v_policy, "interaction": v_interaction,
        }

        print(f"  {agent:14s}  {v_total:10.6f}  {v_env:10.6f}  "
              f"{v_policy:10.6f}  {v_interaction:10.6f}")

    plot_path = Path("results/plots") / f"{name}_variance.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_variance_decomposition(var_table, str(plot_path))
    print(f"\n  Plot -> {plot_path}")

    return var_table

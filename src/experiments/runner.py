"""Experiment runner: trains agents, runs multi-seed episodes, saves results."""

import csv
import numpy as np
from pathlib import Path

from src.env.game_registry import get_game
from src.agents.cfr_agent import CFRAgent
from src.agents.q_learning_agent import QLearningAgent
from src.agents.q_learning_frozen_agent import QLearningFrozenAgent
from src.utils.metrics import (
    summarize_seed, statistical_summary, format_stat_table,
)
from src.utils.plotting import plot_results


def play_episode(env, agent, action_rng, cards, mask_active=None):
    """Play one self-play episode with a predetermined card deal."""
    env.reset(cards=cards, mask_active=mask_active)
    trajectory = []

    while not env.is_terminal:
        player = env.current_player
        info = env.info_state_str(player)
        legal = env.legal_actions()
        action = agent.select_action(info, legal, action_rng)
        trajectory.append((player, info, action))
        env.step(action)

    return env.returns[0], trajectory


# ---------------------------------------------------------------------------
# Config parsing helpers
# ---------------------------------------------------------------------------

def _parse_perturbation(config, action_map):
    """Extract perturbation parameters from config, handling all formats."""
    pert = config["perturbation"]
    disabled = pert.get("disabled", False)
    affected = pert.get("affected_player", 0)
    mask_prob = pert.get("mask_prob", 1.0)

    node_masks_raw = pert.get("node_masks")
    if node_masks_raw is not None:
        node_masks = {}
        for node, actions in node_masks_raw.items():
            node_masks[node] = [action_map[a] for a in actions]
        return disabled, affected, None, False, node_masks, mask_prob

    removed_str = pert.get("removed_action")
    if removed_str and removed_str in action_map:
        removed = action_map[removed_str]
    else:
        default_actions = list(action_map.values())
        removed = default_actions[-1] if default_actions else 1
    root_only = pert.get("root_only", False)
    return disabled, affected, removed, root_only, None, mask_prob


def _make_env(wrapper_class, env_class, removed, affected, root_only,
              node_masks, mask_prob):
    """Create a perturbation-wrapped environment."""
    if node_masks is not None:
        return wrapper_class(
            env_class(), affected_player=affected,
            node_masks=node_masks, mask_prob=mask_prob)
    return wrapper_class(
        env_class(), removed_action=removed,
        affected_player=affected, root_only=root_only,
        mask_prob=mask_prob)


# ---------------------------------------------------------------------------
# Single-seed runner (public: used by variance decomposition)
# ---------------------------------------------------------------------------

def run_single_seed(seed, policy, config, game_info, include_frozen=False,
                    override_env_seed=None, override_policy_seed=None,
                    csv_suffix="", quiet=False):
    """Run one seed. Returns (results_list, csv_path, seed_summary).

    Parameters
    ----------
    seed : int
    policy : dict
    config : dict
    game_info : dict
        Output of ``get_game()`` -- env_class, wrapper_class, deal_fn,
        constants, etc.
    include_frozen : bool
    override_env_seed, override_policy_seed : int or None
    csv_suffix : str
    quiet : bool
    """
    name = config["experiment"]["name"]
    num_episodes = config["experiment"]["num_episodes"]
    perturbation_ep = config["experiment"]["perturbation_episode"]
    alpha = config["q_learning"]["alpha"]
    epsilon = config["q_learning"]["epsilon"]
    num_actions = game_info["constants"]["num_actions"]
    action_map = game_info["constants"]["action_map"]

    disabled, affected, removed, root_only, node_masks, mask_prob = \
        _parse_perturbation(config, action_map)

    # ------------------------------------------------------------------
    # RNG decomposition
    # ------------------------------------------------------------------
    master_rng = np.random.default_rng(seed)
    env_sub_seed = int(master_rng.integers(1 << 63))
    cfr_sub_seed = int(master_rng.integers(1 << 63))
    ql_sub_seed = int(master_rng.integers(1 << 63))
    qlf_sub_seed = int(master_rng.integers(1 << 63))

    env_seed = override_env_seed if override_env_seed is not None else env_sub_seed
    env_rng = np.random.default_rng(env_seed)
    card_rng = np.random.default_rng(env_rng.integers(1 << 63))
    mask_rng = np.random.default_rng(env_rng.integers(1 << 63))

    if override_policy_seed is not None:
        cfr_action_rng = np.random.default_rng(override_policy_seed)
        ql_action_rng = np.random.default_rng(override_policy_seed + 1)
        qlf_action_rng = np.random.default_rng(override_policy_seed + 2)
    else:
        cfr_action_rng = np.random.default_rng(cfr_sub_seed)
        ql_action_rng = np.random.default_rng(ql_sub_seed)
        qlf_action_rng = np.random.default_rng(qlf_sub_seed)

    # ------------------------------------------------------------------
    # Create agents
    # ------------------------------------------------------------------
    cfr_agent = CFRAgent(policy, num_actions=num_actions)
    ql_agent = QLearningAgent(alpha=alpha, epsilon=epsilon,
                              num_actions=num_actions)

    qlf_agent = None
    if include_frozen:
        frozen_cfg = config.get("q_learning_frozen", {})
        qlf_agent = QLearningFrozenAgent(
            alpha=alpha, epsilon=epsilon,
            frozen_epsilon=frozen_cfg.get("frozen_epsilon", 0.0),
            num_actions=num_actions)

    # ------------------------------------------------------------------
    # Create environments
    # ------------------------------------------------------------------
    env_cls = game_info["env_class"]
    wrapper_cls = game_info["wrapper_class"]
    deal_fn = game_info["deal_fn"]

    cfr_env = _make_env(wrapper_cls, env_cls, removed, affected,
                        root_only, node_masks, mask_prob)
    ql_env = _make_env(wrapper_cls, env_cls, removed, affected,
                       root_only, node_masks, mask_prob)
    qlf_env = _make_env(wrapper_cls, env_cls, removed, affected,
                        root_only, node_masks, mask_prob) \
        if include_frozen else None

    if not quiet:
        mode = "disabled" if disabled else (
            "root only" if root_only else "all P0 nodes")
        print(f"\n  Seed {seed}: {num_episodes:,} episodes "
              f"(perturbation at {perturbation_ep:,}, {mode})")

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------
    results = []

    for ep in range(num_episodes):
        if ep == perturbation_ep and not disabled:
            cfr_env.set_perturbed(True)
            ql_env.set_perturbed(True)
            if qlf_env is not None:
                qlf_env.set_perturbed(True)
                qlf_agent.freeze()

        cards = deal_fn(card_rng)

        ep_mask_active = None
        if mask_prob < 1.0 and ep >= perturbation_ep and not disabled:
            ep_mask_active = bool(mask_rng.random() < mask_prob)

        cfr_reward, _ = play_episode(
            cfr_env, cfr_agent, cfr_action_rng, cards, ep_mask_active)
        results.append((ep, cfr_reward, "CFR"))

        ql_reward, ql_traj = play_episode(
            ql_env, ql_agent, ql_action_rng, cards, ep_mask_active)
        ql_agent.update(ql_traj, ql_reward)
        results.append((ep, ql_reward, "Q-Learning"))

        if qlf_agent is not None:
            qlf_reward, qlf_traj = play_episode(
                qlf_env, qlf_agent, qlf_action_rng, cards, ep_mask_active)
            qlf_agent.update(qlf_traj, qlf_reward)
            results.append((ep, qlf_reward, "QL-Frozen"))

    # ------------------------------------------------------------------
    # Summary and CSV
    # ------------------------------------------------------------------
    seed_summary = summarize_seed(results, perturbation_ep, num_episodes)

    if not quiet:
        agents_present = sorted(set(r[2] for r in results))
        finals = {}
        for a in agents_present:
            finals[a] = np.mean([r[1] for r in results[-2000:] if r[2] == a])
        parts = "  ".join(f"{a}: {v:+.3f}" for a, v in finals.items())
        print(f"    done  |  {parts}")

    csv_path = Path("results/raw") / f"{name}{csv_suffix}_seed{seed}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "agent"])
        writer.writerows(results)

    capacity = cfr_env.decision_capacity if not disabled else \
        game_info["constants"]["num_actions"]
    seed_summary["_meta"] = {"capacity": capacity, "seed": seed}

    return results, str(csv_path), seed_summary


# ---------------------------------------------------------------------------
# Full experiment (multi-seed)
# ---------------------------------------------------------------------------

def run_experiment(config):
    """Run a full multi-seed experiment.

    Dispatches on ``config["experiment"]["game"]`` (default: "kuhn").
    Deterministic components (CFR training) run once. Stochastic components
    (card deals, policy execution) run per-seed with controlled RNG streams.

    Returns (csv_paths, plot_path, stat_summary).
    """
    name = config["experiment"]["name"]
    seeds = config["experiment"]["seeds"]
    cfr_iters = config["cfr"]["iterations"]
    game_name = config["experiment"].get("game", "kuhn")
    include_frozen = "q_learning_frozen" in config

    game_info = get_game(game_name)
    constants = game_info["constants"]

    # --- CFR training: deterministic, run once ---
    print(f"\n{'=' * 60}")
    print(f"  Experiment: {name}  ({len(seeds)} seeds, game={game_name})")
    print(f"{'=' * 60}")
    print(f"Training CFR ({cfr_iters:,} iterations) -- deterministic, "
          f"shared across seeds...")

    trainer = game_info["cfr_trainer_class"]()
    trainer.train(cfr_iters)
    policy = trainer.get_average_strategy()

    nash_val = constants.get("nash_value_p0")
    if nash_val is None and hasattr(trainer, "nash_value_p0"):
        nash_val = trainer.nash_value_p0()
        constants["nash_value_p0"] = nash_val

    n_info = len(policy)
    print(f"  CFR converged: {n_info} information sets")
    if nash_val is not None:
        print(f"  Nash value (P0): {nash_val:+.6f}")

    if include_frozen:
        frozen_eps = config["q_learning_frozen"].get("frozen_epsilon", 0.0)
        print(f"  QL-Frozen enabled (frozen_epsilon={frozen_eps})")

    # --- Per-seed runs ---
    csv_paths = []
    seed_summaries = []
    all_results = []

    for seed in seeds:
        results, csv_path, seed_summary = run_single_seed(
            seed, policy, config, game_info,
            include_frozen=include_frozen)
        csv_paths.append(csv_path)
        seed_summaries.append(seed_summary)
        all_results.append(results)

    # --- Aggregated statistics ---
    clean_summaries = [{k: v for k, v in s.items() if k != "_meta"}
                       for s in seed_summaries]
    stat_summary = statistical_summary(clean_summaries)

    meta = seed_summaries[0].get("_meta", {})
    stat_summary["_meta"] = {
        "capacity": meta.get("capacity", -1),
        "game": game_name,
        "constants": constants,
    }

    print(f"\n--- Statistical Summary ({len(seeds)} seeds) ---")
    print(format_stat_table(stat_summary))

    # --- Plot (aggregated across seeds) ---
    plot_path = Path("results/plots") / f"{name}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_results(csv_paths, config, str(plot_path))
    print(f"\nPlot -> {plot_path}")

    return csv_paths, str(plot_path), stat_summary

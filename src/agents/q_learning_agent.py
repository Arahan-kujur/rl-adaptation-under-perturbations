"""Tabular Q-learning agent -- game-agnostic."""

import numpy as np
from collections import defaultdict


class QLearningAgent:
    """Epsilon-greedy tabular Q-learning with MC-style terminal updates.

    Uses terminal reward directly (no discounting) since poker
    episodes are only a few actions long.
    """

    def __init__(self, alpha=0.1, epsilon=0.15, num_actions=2):
        self.num_actions = num_actions
        self.q = defaultdict(lambda: np.zeros(self.num_actions))
        self.alpha = alpha
        self.epsilon = epsilon

    def select_action(self, info_state, legal_actions, rng):
        if rng.random() < self.epsilon:
            return int(rng.choice(legal_actions))
        q_vals = self.q[info_state].copy()
        masked = np.full(self.num_actions, -np.inf)
        for a in legal_actions:
            masked[a] = q_vals[a]
        return int(np.argmax(masked))

    def update(self, trajectory, reward_p0):
        for player, info_state, action in trajectory:
            r = reward_p0 if player == 0 else -reward_p0
            self.q[info_state][action] += self.alpha * (r - self.q[info_state][action])

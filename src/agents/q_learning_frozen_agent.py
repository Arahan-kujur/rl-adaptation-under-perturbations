"""Tabular Q-learning agent that freezes learning after perturbation."""

from src.agents.q_learning_agent import QLearningAgent


class QLearningFrozenAgent(QLearningAgent):
    """Q-learning that stops updating after freeze() is called.

    Models a deployed policy: trained during normal play, then held fixed
    when the environment changes. Optionally decays epsilon to a configured
    value upon freezing, allowing pure exploitation of the learned Q-table.
    """

    def __init__(self, alpha=0.1, epsilon=0.15, frozen_epsilon=0.0,
                 num_actions=2):
        super().__init__(alpha=alpha, epsilon=epsilon, num_actions=num_actions)
        self._frozen = False
        self._training_epsilon = epsilon
        self._frozen_epsilon = frozen_epsilon

    @property
    def is_frozen(self):
        return self._frozen

    def freeze(self):
        self._frozen = True
        self.epsilon = self._frozen_epsilon

    def update(self, trajectory, reward_p0):
        if not self._frozen:
            super().update(trajectory, reward_p0)

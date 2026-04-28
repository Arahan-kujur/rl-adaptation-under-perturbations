"""Kuhn Poker environment with configurable asymmetric action perturbation."""

import numpy as np

PASS = 0
BET = 1
NUM_ACTIONS = 2
CARD_NAMES = {0: "J", 1: "Q", 2: "K"}

ACTION_NAMES = {"pass": PASS, "bet": BET}
P0_HISTORY_STRS = ["", "pb"]


class KuhnPokerEnv:
    """Kuhn Poker: 3 cards (J<Q<K), 2 players, ante 1, bet 1.

    Game tree (P=pass, B=bet):
      P0: P or B
        P -> P1: P(showdown +/-1) or B -> P0: P(fold,-1) or B(call, showdown +/-2)
        B -> P1: P(fold,+1) or B(call, showdown +/-2)
    """

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()
        self._cards = [0, 0]
        self._history = []
        self._done = False
        self._reward_p0 = 0

    def reset(self, cards=None):
        if cards is not None:
            self._cards = list(cards)
        else:
            deck = np.array([0, 1, 2])
            self.rng.shuffle(deck)
            self._cards = [int(deck[0]), int(deck[1])]
        self._history = []
        self._done = False
        self._reward_p0 = 0
        return self

    @property
    def current_player(self):
        n = len(self._history)
        if n == 0:
            return 0
        if n == 1:
            return 1
        if n == 2:
            return 0
        return -1

    @property
    def is_root(self):
        return len(self._history) == 0

    @property
    def history_str(self):
        return "".join("p" if a == PASS else "b" for a in self._history)

    def info_state_str(self, player):
        card = self._cards[player]
        return f"{card}{self.history_str}"

    def legal_actions(self):
        return [PASS, BET]

    @property
    def is_terminal(self):
        return self._done

    @property
    def returns(self):
        return [self._reward_p0, -self._reward_p0]

    def step(self, action):
        assert not self._done, "step() called on terminal state"
        self._history.append(action)
        h = tuple(self._history)

        showdown = 1 if self._cards[0] > self._cards[1] else -1
        terminal_payoffs = {
            (PASS, PASS): showdown,
            (BET, PASS): 1,
            (BET, BET): 2 * showdown,
            (PASS, BET, PASS): -1,
            (PASS, BET, BET): 2 * showdown,
        }

        if h in terminal_payoffs:
            self._done = True
            self._reward_p0 = terminal_payoffs[h]


class PerturbedKuhnPoker:
    """Wrapper that filters actions for one player under perturbation.

    Supports three masking modes (checked in priority order):

    1. **node_masks** (dict: history_str -> list of action ints to remove)
       Most general. Allows per-node action removal.

    2. **Legacy** (removed_action + root_only)
       Original interface. ``root_only=False`` strips at all P0 nodes;
       ``root_only=True`` strips only at the opening move.

    3. **disabled** perturbation
       ``set_perturbed(True)`` is never called by the runner when the
       config sets ``disabled: true``, so legal_actions always passes
       through unmodified.

    Additionally, ``mask_prob`` (0-1) controls stochastic masking:
    each episode the runner draws whether the mask is active. When
    ``mask_active=False`` is passed to ``reset()``, the perturbation
    has no effect for that episode even if ``perturbed`` is True.
    """

    def __init__(self, env, removed_action=BET, affected_player=0,
                 root_only=False, node_masks=None, mask_prob=1.0):
        self.env = env
        self.perturbed = False
        self.removed_action = removed_action
        self.affected_player = affected_player
        self.root_only = root_only
        self.node_masks = node_masks
        self.mask_prob = mask_prob
        self._mask_active = True

    def set_perturbed(self, flag):
        self.perturbed = flag

    def reset(self, cards=None, mask_active=None):
        self.env.reset(cards=cards)
        self._mask_active = mask_active if mask_active is not None else True
        return self

    @property
    def current_player(self):
        return self.env.current_player

    def info_state_str(self, player):
        return self.env.info_state_str(player)

    def legal_actions(self):
        actions = self.env.legal_actions()
        if not (self.perturbed and self._mask_active):
            return actions
        if self.env.current_player != self.affected_player:
            return actions

        if self.node_masks is not None:
            h = self.env.history_str
            if h in self.node_masks:
                filtered = [a for a in actions if a not in self.node_masks[h]]
                if filtered:
                    return filtered
        else:
            if not self.root_only or self.env.is_root:
                filtered = [a for a in actions if a != self.removed_action]
                if filtered:
                    return filtered
        return actions

    @property
    def decision_capacity(self):
        """Number of P0 decision points retaining >1 legal action.

        Computed from the mask configuration, not from runtime state.
        For stochastic masking this reports the capacity when the mask
        is active (worst case).
        """
        if not self.perturbed:
            return 2

        all_actions = [PASS, BET]
        if self.node_masks is not None:
            count = 0
            for h in P0_HISTORY_STRS:
                if h in self.node_masks:
                    remaining = [a for a in all_actions
                                 if a not in self.node_masks[h]]
                    count += int(len(remaining) > 1)
                else:
                    count += 1
            return count

        if not self.root_only:
            return 0
        return 1

    @property
    def is_terminal(self):
        return self.env.is_terminal

    @property
    def returns(self):
        return self.env.returns

    def step(self, action):
        self.env.step(action)

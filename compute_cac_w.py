"""Compute reach-weighted CAC for Kuhn Poker perturbation conditions."""

import numpy as np

# Known Kuhn Nash strategy (approximate)
# P0 at root (info state = card only): [P(pass), P(bet)]
P0_ROOT = {
    "J": [0.78, 0.22],
    "Q": [1.00, 0.00],
    "K": [0.33, 0.67],
}

# P1 after P0 passes (info state = card + "p"): [P(pass), P(bet)]
P1_AFTER_PASS = {
    "J": [0.67, 0.33],
    "Q": [0.44, 0.56],
    "K": [0.00, 1.00],
}

# All 6 card deals (P0_card, P1_card), each equally likely
DEALS = [
    ("J", "Q"), ("J", "K"),
    ("Q", "J"), ("Q", "K"),
    ("K", "J"), ("K", "Q"),
]


def reach_pb_node(p0_card, p1_card):
    """Reach probability of the 'pb' node (P0 passed, then P1 bet)."""
    p0_pass_prob = P0_ROOT[p0_card][0]
    p1_bet_prob = P1_AFTER_PASS[p1_card][1]
    return p0_pass_prob * p1_bet_prob


def reach_root_node(p0_card, p1_card):
    """Reach probability of P0's root decision node (always 1 under Nash)."""
    return 1.0


def main():
    n_deals = len(DEALS)

    # Full removal (CAC=0): all P0 nodes removed, so CAC_w = 0
    cac_w_full_removal = 0.0

    # Root-only (CAC=1): only the "pb" node remains as a choice node.
    # Reach of "pb" node averaged over deals.
    pb_reaches = [reach_pb_node(c0, c1) for c0, c1 in DEALS]
    cac_w_root_only = np.mean(pb_reaches)

    # No perturbation (CAC=2): both P0 nodes (root + pb) are choice nodes.
    # Sum of reaches for each deal, then average across deals.
    total_reaches = []
    for c0, c1 in DEALS:
        root_reach = reach_root_node(c0, c1)
        pb_reach = reach_pb_node(c0, c1)
        total_reaches.append(root_reach + pb_reach)
    cac_w_no_perturbation = np.mean(total_reaches)

    print(f"{'Perturbation':<20s}{'CAC (unweighted)':<20s}{'CAC_w (reach-weighted)'}")
    print("-" * 60)
    print(f"{'Full removal':<20s}{0:<20d}{cac_w_full_removal:.3f}")
    print(f"{'Root-only':<20s}{1:<20d}{cac_w_root_only:.3f}")
    print(f"{'No perturbation':<20s}{2:<20d}{cac_w_no_perturbation:.3f}")

    print("\n--- Detail: per-deal reach of 'pb' node ---")
    for (c0, c1), r in zip(DEALS, pb_reaches):
        print(f"  Deal {c0}{c1}: P0 pass={P0_ROOT[c0][0]:.2f}, "
              f"P1 bet={P1_AFTER_PASS[c1][1]:.2f}, reach(pb)={r:.4f}")


if __name__ == "__main__":
    main()

"""Autonomy Asymmetry Index (AAI) — Definition 3 from the spec.

AAI(i, ε) = E(i, ε) / S(i, ε)

S(i, ε) = self-influence displacement: max Jaccard displacement of user i's
           top-k achievable by perturbing their OWN ratings with budget ε.
E(i, ε) = max adversary displacement: max Jaccard displacement achievable by
           ANY adversary perturbing THEIR ratings with budget ε.

AAI < 1: healthy (user has more power than strangers)
AAI = 1: borderline
AAI > 1: problematic (strangers have more power than user)
"""

import numpy as np
from causallens.recommender import Recommender
from causallens.metrics.manipulation import manipulation_resistance


def _jaccard_distance(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def self_influence(
    recommender: Recommender,
    user_id: int,
    rating_matrix: np.ndarray,
    k: int = 10,
    epsilon: int = 10,
    n_trials: int = 10,
) -> float:
    """Compute S(i, ε) — self-influence displacement.

    Measures how much a user can shift their own top-k by changing up to
    ε of their own ratings. Uses greedy hill-climbing: at each step, try
    changing one unrated→rated or modifying an existing rating to maximise
    Jaccard displacement.

    Args:
        recommender: Recommender instance.
        user_id: User index.
        rating_matrix: Full rating matrix (n × m).
        k: Top-k list size.
        epsilon: Budget — max number of rating changes.
        n_trials: Number of random restarts for robustness.

    Returns:
        Float in [0, 1] — max Jaccard displacement achieved.
    """
    orig_topk = set(
        recommender.get_recommendations(user_id, rating_matrix, k).tolist()
    )
    m = rating_matrix.shape[1]
    rating_values = [1.0, 2.0, 3.0, 4.0, 5.0]

    best_displacement = 0.0

    for _ in range(n_trials):
        R = rating_matrix.copy()

        for step in range(epsilon):
            # Try changing one rating to maximise displacement
            candidates = np.where(R[user_id] == 0)[0]
            if len(candidates) > 200:
                candidates = np.random.choice(candidates, 200, replace=False)

            best_step_disp = 0.0
            best_item = -1
            best_val = 0.0

            for item in candidates:
                old_val = R[user_id, item]
                for val in rating_values:
                    R[user_id, item] = val
                    scores = recommender.get_scores(user_id, R)
                    scores_masked = scores.copy()
                    scores_masked[R[user_id] > 0] = -np.inf
                    new_topk = set(np.argsort(scores_masked)[::-1][:k].tolist())
                    disp = _jaccard_distance(orig_topk, new_topk)
                    if disp > best_step_disp:
                        best_step_disp = disp
                        best_item = item
                        best_val = val
                R[user_id, item] = old_val

            if best_item == -1:
                break

            R[user_id, best_item] = best_val

            if best_step_disp > best_displacement:
                best_displacement = best_step_disp

    return best_displacement


def autonomy_asymmetry_index(
    recommender: Recommender,
    user_id: int,
    rating_matrix: np.ndarray,
    k: int = 10,
    epsilon: int = 10,
    adversary_ids: list[int] | None = None,
    n_adversaries: int = 3,
    mode: str = "blackbox",
    self_trials: int = 5,
) -> dict:
    """Compute AAI(i, ε) = E(i, ε) / S(i, ε).

    Args:
        recommender: Recommender instance.
        user_id: User index.
        rating_matrix: Full rating matrix.
        k: Top-k list size.
        epsilon: Budget for both self and adversary perturbations.
        adversary_ids: Specific adversaries to test. If None, sample randomly.
        n_adversaries: Number of adversaries if sampling.
        mode: "whitebox" or "blackbox" for adversary computation.
        self_trials: Random restarts for self-influence.

    Returns:
        dict with aai, self_influence, max_adversary_displacement, label.
    """
    # S(i, ε): self-influence
    s = self_influence(
        recommender, user_id, rating_matrix, k, epsilon, n_trials=self_trials
    )

    # E(i, ε): max adversary displacement
    if adversary_ids is None:
        all_others = [u for u in range(rating_matrix.shape[0]) if u != user_id]
        adversary_ids = np.random.choice(
            all_others,
            min(n_adversaries, len(all_others)),
            replace=False,
        ).tolist()

    max_adv_disp = 0.0
    for aid in adversary_ids:
        result = manipulation_resistance(
            recommender, user_id, aid, rating_matrix,
            k=k, epsilon=epsilon, mode=mode,
        )
        if result["displacement"] > max_adv_disp:
            max_adv_disp = result["displacement"]

    # AAI = E / S (guard against division by zero)
    if s < 1e-10:
        # User has zero self-influence — any adversary influence is infinite asymmetry
        aai = float("inf") if max_adv_disp > 0 else 1.0
    else:
        aai = max_adv_disp / s

    # Label
    if aai < 1.0:
        label = "healthy"
    elif aai == 1.0:
        label = "borderline"
    else:
        label = "problematic"

    return {
        "aai": aai,
        "self_influence": s,
        "adversary_influence": max_adv_disp,
        "epsilon": epsilon,
        "label": label,
        "user_id": user_id,
    }

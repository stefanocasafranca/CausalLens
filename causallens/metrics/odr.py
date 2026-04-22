"""Observational Deception Rate (ODR) — Definition 4.

Of users who LOOK autonomous by observational metrics (diversity, coverage,
volatility), what percentage are actually trapped when tested causally.

ODR = count(deceptive) / count(observationally flagged as autonomous)

A user is "observationally autonomous" if they score strictly above the
population median on diversity AND volatility.

A user is "causally trapped" if their reachability success rate is zero —
meaning none of the target items could be pushed into top-k within budget.

"Deceptive" = observationally autonomous AND causally trapped.
"""

import numpy as np
from causallens.recommender import Recommender
from causallens.metrics.observational import (
    intra_list_diversity,
    recommendation_volatility,
)
from causallens.metrics.reachability import reachability_cost


def observational_deception_rate(
    recommender: Recommender,
    user_ids: list[int],
    rating_matrix: np.ndarray,
    k: int = 10,
    max_budget: int = 20,
    n_targets_per_user: int = 3,
    reachability_mode: str = "blackbox",
) -> dict:
    """Compute ODR for a set of users.

    Steps:
    1. Compute observational metrics (diversity, volatility) for all users.
    2. Flag users strictly above population median on BOTH as "observationally autonomous".
    3. Compute reachability cost for flagged users.
    4. Users with zero reachability successes (all targets unreachable) are "causally trapped".
    5. ODR = count(trapped AND flagged) / count(flagged).

    Args:
        recommender: Recommender instance.
        user_ids: List of user indices.
        rating_matrix: Full rating matrix.
        k: Top-k list size.
        max_budget: Max budget for reachability computation.
        n_targets_per_user: Target items per user for reachability.
        reachability_mode: "whitebox" or "blackbox".

    Returns:
        dict with odr, counts, per-user details.
    """
    # Step 1: Observational metrics
    diversities = {}
    volatilities = {}
    for uid in user_ids:
        diversities[uid] = intra_list_diversity(recommender, uid, rating_matrix, k)
        volatilities[uid] = recommendation_volatility(recommender, uid, rating_matrix, k)

    div_values = np.array(list(diversities.values()))
    vol_values = np.array(list(volatilities.values()))
    div_median = float(np.median(div_values))
    vol_median = float(np.median(vol_values))

    # Step 2: Flag observationally autonomous
    flagged = [
        uid for uid in user_ids
        if diversities[uid] > div_median and volatilities[uid] > vol_median
    ]

    if len(flagged) == 0:
        return {
            "odr": 0.0,
            "n_flagged": 0,
            "n_deceptive": 0,
            "n_total": len(user_ids),
            "div_median": div_median,
            "vol_median": vol_median,
            "details": [],
        }

    # Step 3: Reachability for flagged users
    reach_successes = {}
    reach_costs = {}
    for uid in flagged:
        current_topk = set(
            recommender.get_recommendations(uid, rating_matrix, k).tolist()
        )
        unrated = np.where(rating_matrix[uid] == 0)[0]
        available = [i for i in unrated if i not in current_topk]
        if len(available) == 0:
            reach_successes[uid] = 0
            reach_costs[uid] = max_budget
            continue

        targets = np.random.choice(
            available,
            min(n_targets_per_user, len(available)),
            replace=False,
        )

        costs = []
        successes = 0
        for target in targets:
            result = reachability_cost(
                recommender, uid, int(target), rating_matrix,
                k=k, max_budget=max_budget, mode=reachability_mode,
            )
            costs.append(result["cost"])
            if result["cost"] < max_budget:
                successes += 1
        reach_successes[uid] = successes
        reach_costs[uid] = float(np.mean(costs))

    # Step 4: Causally trapped = zero reachability successes
    # "Deceptive" = flagged as autonomous BUT causally trapped
    deceptive = [uid for uid in flagged if reach_successes[uid] == 0]

    # Step 5: ODR
    odr = len(deceptive) / len(flagged)

    details = []
    for uid in flagged:
        details.append({
            "user_id": uid,
            "diversity": diversities[uid],
            "volatility": volatilities[uid],
            "mean_reachability_cost": reach_costs[uid],
            "reachability_successes": reach_successes[uid],
            "is_deceptive": uid in deceptive,
        })

    return {
        "odr": odr,
        "n_flagged": len(flagged),
        "n_deceptive": len(deceptive),
        "n_total": len(user_ids),
        "div_median": div_median,
        "vol_median": vol_median,
        "details": details,
    }

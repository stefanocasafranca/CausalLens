"""Observational autonomy baselines — non-causal metrics.

These metrics measure surface-level properties of recommendations that
*look like* autonomy but can be deceptive (see ODR, Definition 4).

1. Intra-list diversity: average pairwise Jaccard distance within top-k
2. Catalog coverage: fraction of total catalog appearing in top-k across users
3. Recommendation volatility: Jaccard distance after small random perturbation
"""

import numpy as np
from causallens.recommender import Recommender


def _jaccard_distance_items(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard distance between two item sets represented as arrays."""
    sa, sb = set(a.tolist()), set(b.tolist())
    if not sa and not sb:
        return 0.0
    return 1.0 - len(sa & sb) / len(sa | sb)


def intra_list_diversity(
    recommender: Recommender,
    user_id: int,
    rating_matrix: np.ndarray,
    k: int = 10,
) -> float:
    """Average pairwise Jaccard distance among items in a user's top-k.

    For each pair of items (a, b) in the top-k, compute the Jaccard distance
    between their "user profiles" — the sets of users who rated each item.
    High diversity = items appeal to different user populations.

    Args:
        recommender: Recommender instance.
        user_id: User index.
        rating_matrix: Full rating matrix (n × m).
        k: Top-k list size.

    Returns:
        Float in [0, 1]. Higher = more diverse recommendations.
    """
    topk = recommender.get_recommendations(user_id, rating_matrix, k)
    if len(topk) < 2:
        return 0.0

    # For each item, get the set of users who rated it
    distances = []
    for i in range(len(topk)):
        users_i = set(np.where(rating_matrix[:, topk[i]] > 0)[0].tolist())
        for j in range(i + 1, len(topk)):
            users_j = set(np.where(rating_matrix[:, topk[j]] > 0)[0].tolist())
            if not users_i and not users_j:
                d = 0.0
            else:
                d = 1.0 - len(users_i & users_j) / len(users_i | users_j)
            distances.append(d)

    return float(np.mean(distances))


def catalog_coverage(
    recommender: Recommender,
    user_ids: list[int],
    rating_matrix: np.ndarray,
    k: int = 10,
) -> float:
    """Fraction of the item catalog appearing in any user's top-k.

    Args:
        recommender: Recommender instance.
        user_ids: List of user indices to evaluate.
        rating_matrix: Full rating matrix (n × m).
        k: Top-k list size.

    Returns:
        Float in [0, 1]. Higher = recommender exposes more of the catalog.
    """
    n_items = rating_matrix.shape[1]
    seen_items = set()
    for uid in user_ids:
        topk = recommender.get_recommendations(uid, rating_matrix, k)
        seen_items.update(topk.tolist())
    return len(seen_items) / n_items


def recommendation_volatility(
    recommender: Recommender,
    user_id: int,
    rating_matrix: np.ndarray,
    k: int = 10,
    n_perturbations: int = 5,
    perturbation_size: int = 1,
) -> float:
    """Jaccard distance between top-k before/after small random perturbation.

    Adds a single random rating (or `perturbation_size` ratings) to previously
    unrated items and measures how much top-k changes. Averaged over
    `n_perturbations` trials.

    High volatility = recommendations are sensitive to small input changes,
    which *looks like* the user has control (but may be deceptive).

    Args:
        recommender: Recommender instance.
        user_id: User index.
        rating_matrix: Full rating matrix (n × m).
        k: Top-k list size.
        n_perturbations: Number of random trials to average.
        perturbation_size: Number of new ratings added per trial.

    Returns:
        Float in [0, 1]. Higher = more volatile (responsive to changes).
    """
    orig_topk = recommender.get_recommendations(user_id, rating_matrix, k)
    unrated = np.where(rating_matrix[user_id] == 0)[0]

    if len(unrated) < perturbation_size:
        return 0.0

    distances = []
    for _ in range(n_perturbations):
        items_to_rate = np.random.choice(unrated, perturbation_size, replace=False)
        old_vals = rating_matrix[user_id, items_to_rate].copy()
        try:
            for item in items_to_rate:
                rating_matrix[user_id, item] = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0])
            new_topk = recommender.get_recommendations(user_id, rating_matrix, k)
            distances.append(_jaccard_distance_items(orig_topk, new_topk))
        finally:
            rating_matrix[user_id, items_to_rate] = old_vals

    return float(np.mean(distances))


def compute_observational_metrics(
    recommender: Recommender,
    user_ids: list[int],
    rating_matrix: np.ndarray,
    k: int = 10,
) -> dict:
    """Compute all three observational metrics for a set of users.

    Returns:
        dict with per-user arrays and summary statistics for each metric.
    """
    diversities = []
    volatilities = []

    for uid in user_ids:
        diversities.append(intra_list_diversity(recommender, uid, rating_matrix, k))
        volatilities.append(recommendation_volatility(recommender, uid, rating_matrix, k))

    coverage = catalog_coverage(recommender, user_ids, rating_matrix, k)

    return {
        "diversity": np.array(diversities),
        "volatility": np.array(volatilities),
        "coverage": coverage,
        "mean_diversity": float(np.mean(diversities)),
        "mean_volatility": float(np.mean(volatilities)),
        "user_ids": user_ids,
    }

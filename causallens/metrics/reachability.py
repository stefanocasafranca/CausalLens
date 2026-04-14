"""Reachability Cost metric — Definition 1 from the spec.

R(i, j, k) = min_δ ‖δ‖₁  s.t. rank(f, j, i | r_i + δ) ≤ k, δ ∈ C

White-box: projected gradient descent on the differentiable scoring path.
Black-box: coordinate-wise finite differences + greedy search.
"""

import numpy as np
import torch


def _jaccard_distance(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def reachability_cost_whitebox(
    recommender,
    user_id: int,
    target_item: int,
    rating_matrix: np.ndarray,
    k: int = 10,
    max_budget: int = 20,
    lr: float = 0.5,
    n_steps: int = 100,
) -> dict:
    """White-box reachability via projected gradient descent.

    Minimises ‖δ‖₁ such that target_item enters top-k.
    Uses the recommender's differentiable scoring path.

    Args:
        recommender: A Recommender with get_scores_differentiable().
        user_id: User index.
        target_item: Item index to push into top-k.
        rating_matrix: The original rating matrix.
        k: Top-k list size.
        max_budget: Maximum ‖δ‖₁ to try.
        lr: Learning rate for gradient descent.
        n_steps: Optimisation steps.

    Returns:
        dict with cost (L1 norm of δ), perturbation, success flag, rank.
    """
    m = rating_matrix.shape[1]
    r_orig = torch.tensor(rating_matrix[user_id], dtype=torch.float32)

    # Items the user hasn't rated — perturbation candidates
    unrated = (r_orig == 0)

    delta = torch.zeros(m, requires_grad=True)

    optimizer = torch.optim.Adam([delta], lr=lr)

    best_cost = float("inf")
    best_delta = None
    success = False

    for step in range(n_steps):
        optimizer.zero_grad()

        # Apply perturbation: clamp ratings to [1, 5] range for rated items
        r_pert = r_orig + delta
        # Only allow perturbation on unrated items (new ratings in [1,5])
        # or small adjustments to rated items
        r_clamped = torch.clamp(r_pert, 0.0, 5.0)

        scores = recommender.get_scores_differentiable(user_id, r_clamped)

        # Loss: push target_item score above k-th highest score
        topk_vals, _ = torch.topk(scores, k)
        threshold = topk_vals[-1]  # k-th highest score
        target_score = scores[target_item]

        # Hinge loss: penalise when target is below threshold
        rank_loss = torch.relu(threshold - target_score + 0.1)
        # L1 regularisation to minimise perturbation
        l1_loss = delta.abs().sum()

        loss = rank_loss + 0.01 * l1_loss
        loss.backward()
        optimizer.step()

        # Project: enforce budget constraint
        with torch.no_grad():
            if delta.abs().sum() > max_budget:
                delta.data = delta.data * (max_budget / delta.abs().sum())

        # Check if target is now in top-k
        with torch.no_grad():
            r_check = torch.clamp(r_orig + delta, 0.0, 5.0)
            s = recommender.get_scores_differentiable(user_id, r_check)
            rank = (s > s[target_item]).sum().item() + 1
            cost = delta.abs().sum().item()
            if rank <= k and cost < best_cost:
                best_cost = cost
                best_delta = delta.detach().clone()
                success = True

    if best_delta is None:
        best_delta = delta.detach()
        best_cost = delta.abs().sum().item()
        with torch.no_grad():
            r_check = torch.clamp(r_orig + best_delta, 0.0, 5.0)
            s = recommender.get_scores_differentiable(user_id, r_check)
            rank = (s > s[target_item]).sum().item() + 1

    return {
        "cost": best_cost,
        "perturbation": best_delta.numpy(),
        "success": success,
        "rank": rank,
        "n_changed": int((best_delta.abs() > 0.01).sum()),
    }


def reachability_cost_blackbox(
    recommender,
    user_id: int,
    target_item: int,
    rating_matrix: np.ndarray,
    k: int = 10,
    max_budget: int = 20,
    rating_values: list[float] | None = None,
) -> dict:
    """Black-box reachability via greedy coordinate search.

    At each step, try adding/changing one rating to maximally improve
    the target item's rank. Stop when it enters top-k or budget exhausted.

    Args:
        recommender: A Recommender with get_scores().
        user_id: User index.
        target_item: Item index to push into top-k.
        rating_matrix: The original rating matrix.
        k: Top-k list size.
        max_budget: Maximum number of rating changes.
        rating_values: Discrete rating values to try (default [1,2,3,4,5]).

    Returns:
        dict with cost, success, rank, changes made.
    """
    if rating_values is None:
        rating_values = [1.0, 2.0, 3.0, 4.0, 5.0]

    R = rating_matrix.copy()
    m = R.shape[1]
    changes = []

    for step in range(max_budget):
        scores = recommender.get_scores(user_id, R)
        # Mask already-rated items for ranking
        rated_mask = R[user_id] > 0
        scores_masked = scores.copy()
        scores_masked[rated_mask] = -np.inf
        rank = int((scores_masked > scores_masked[target_item]).sum()) + 1

        if rank <= k:
            return {
                "cost": len(changes),
                "success": True,
                "rank": rank,
                "changes": changes,
            }

        # Greedy: try each unrated item × rating value, pick best
        best_rank = rank
        best_item = -1
        best_val = 0.0

        # Candidate items: unrated items (excluding target)
        candidates = np.where(R[user_id] == 0)[0]
        candidates = candidates[candidates != target_item]

        # Sample if too many candidates
        if len(candidates) > 200:
            candidates = np.random.choice(candidates, 200, replace=False)

        for item in candidates:
            for val in rating_values:
                R_try = R.copy()
                R_try[user_id, item] = val
                s = recommender.get_scores(user_id, R_try)
                s_masked = s.copy()
                mask = R_try[user_id] > 0
                s_masked[mask] = -np.inf
                r = int((s_masked > s_masked[target_item]).sum()) + 1
                if r < best_rank:
                    best_rank = r
                    best_item = item
                    best_val = val

        if best_item == -1:
            break  # No improvement possible

        R[user_id, best_item] = best_val
        changes.append((best_item, best_val))

    # Final check
    scores = recommender.get_scores(user_id, R)
    rated_mask = R[user_id] > 0
    scores[rated_mask] = -np.inf
    rank = int((scores > scores[target_item]).sum()) + 1

    return {
        "cost": len(changes),
        "success": rank <= k,
        "rank": rank,
        "changes": changes,
    }


def reachability_cost(
    recommender,
    user_id: int,
    target_item: int,
    rating_matrix: np.ndarray,
    k: int = 10,
    max_budget: int = 20,
    mode: str = "whitebox",
    **kwargs,
) -> dict:
    """Compute reachability cost R(i, j, k).

    Args:
        mode: "whitebox" (requires differentiable scoring) or "blackbox".
    """
    if mode == "whitebox":
        return reachability_cost_whitebox(
            recommender, user_id, target_item, rating_matrix, k, max_budget, **kwargs
        )
    elif mode == "blackbox":
        return reachability_cost_blackbox(
            recommender, user_id, target_item, rating_matrix, k, max_budget, **kwargs
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'whitebox' or 'blackbox'.")

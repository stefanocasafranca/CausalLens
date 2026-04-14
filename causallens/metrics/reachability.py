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
    lr: float = 1.0,
    n_steps: int = 100,
) -> dict:
    """White-box reachability via projected gradient descent.

    Two-phase approach:
    Phase 1: Maximise target item score to push it into top-k (ignore cost).
    Phase 2: If successful, binary search on budget to find minimum cost.

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

    best_cost = float("inf")
    best_delta = None
    success = False
    final_rank = m

    # Phase 1: push target into top-k using full budget
    delta = torch.zeros(m, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()
        r_pert = torch.clamp(r_orig + delta, 0.0, 5.0)
        scores = recommender.get_scores_differentiable(user_id, r_pert)

        # Direct loss: maximise target score, minimise top-k scores
        target_score = scores[target_item]
        # Get k-th highest score excluding target
        scores_no_target = scores.clone()
        scores_no_target[target_item] = -1e9
        topk_vals, _ = torch.topk(scores_no_target, k)
        threshold = topk_vals[-1]

        # Maximise gap: target_score - threshold
        loss = -(target_score - threshold)
        loss.backward()
        optimizer.step()

        # Project onto L1 ball
        with torch.no_grad():
            if delta.abs().sum() > max_budget:
                delta.data = _project_l1_reachability(delta.data, max_budget)

        # Check rank
        with torch.no_grad():
            r_check = torch.clamp(r_orig + delta, 0.0, 5.0)
            s = recommender.get_scores_differentiable(user_id, r_check)
            rank = (s > s[target_item]).sum().item() + 1
            cost = delta.abs().sum().item()
            if rank <= k and cost < best_cost:
                best_cost = cost
                best_delta = delta.detach().clone()
                success = True
                final_rank = rank

    # Phase 2: if successful, try with smaller budgets via binary search
    if success:
        lo, hi = 0.0, best_cost
        for _ in range(5):
            mid = (lo + hi) / 2
            delta2 = torch.zeros(m, requires_grad=True)
            opt2 = torch.optim.Adam([delta2], lr=lr)
            found = False
            for step in range(n_steps // 2):
                opt2.zero_grad()
                r_pert = torch.clamp(r_orig + delta2, 0.0, 5.0)
                scores = recommender.get_scores_differentiable(user_id, r_pert)
                target_score = scores[target_item]
                scores_no_target = scores.clone()
                scores_no_target[target_item] = -1e9
                topk_vals, _ = torch.topk(scores_no_target, k)
                threshold = topk_vals[-1]
                loss = -(target_score - threshold)
                loss.backward()
                opt2.step()
                with torch.no_grad():
                    if delta2.abs().sum() > mid:
                        delta2.data = _project_l1_reachability(delta2.data, mid)
                    r_check = torch.clamp(r_orig + delta2, 0.0, 5.0)
                    s = recommender.get_scores_differentiable(user_id, r_check)
                    rank = (s > s[target_item]).sum().item() + 1
                    if rank <= k:
                        found = True
                        c = delta2.abs().sum().item()
                        if c < best_cost:
                            best_cost = c
                            best_delta = delta2.detach().clone()
                            final_rank = rank
                        break
            if found:
                hi = mid
            else:
                lo = mid

    if best_delta is None:
        best_delta = delta.detach()
        best_cost = delta.abs().sum().item()
        final_rank = m
        with torch.no_grad():
            r_check = torch.clamp(r_orig + best_delta, 0.0, 5.0)
            s = recommender.get_scores_differentiable(user_id, r_check)
            final_rank = (s > s[target_item]).sum().item() + 1

    return {
        "cost": best_cost,
        "perturbation": best_delta.numpy(),
        "success": success,
        "rank": final_rank,
        "n_changed": int((best_delta.abs() > 0.01).sum()),
    }


def _project_l1_reachability(x: torch.Tensor, radius: float) -> torch.Tensor:
    """Project x onto the L1 ball of given radius."""
    if x.abs().sum() <= radius:
        return x
    u = x.abs().sort(descending=True).values
    cumsum = torch.cumsum(u, dim=0)
    rho = torch.where(u > (cumsum - radius) / torch.arange(1, len(u) + 1, dtype=x.dtype))[0]
    if len(rho) == 0:
        return torch.zeros_like(x)
    rho = rho[-1]
    theta = max(0, (cumsum[rho] - radius) / (rho + 1))
    return torch.sign(x) * torch.clamp(x.abs() - theta, min=0)


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
            old_val = R[user_id, item]
            for val in rating_values:
                R[user_id, item] = val
                s = recommender.get_scores(user_id, R)
                s_masked = s.copy()
                mask = R[user_id] > 0
                s_masked[mask] = -np.inf
                r = int((s_masked > s_masked[target_item]).sum()) + 1
                if r < best_rank:
                    best_rank = r
                    best_item = item
                    best_val = val
            R[user_id, item] = old_val

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

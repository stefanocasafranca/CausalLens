"""Manipulation Resistance metric — Definition 2 from the spec.

M(i, a, ε) = max_{δ_a} d(top-k(f,i|R), top-k(f,i|R+Δ_a))
    s.t. ‖δ_a‖₁ ≤ ε, Δ_a modifies ONLY adversary a's ratings.

White-box: gradient ascent on Jaccard displacement.
Black-box: hill-climbing over adversary rating changes.
"""

import numpy as np
import torch


def _jaccard_distance(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def manipulation_resistance_whitebox(
    recommender,
    victim_id: int,
    adversary_id: int,
    rating_matrix: np.ndarray,
    k: int = 10,
    epsilon: int = 10,
    lr: float = 0.5,
    n_steps: int = 80,
) -> dict:
    """White-box manipulation resistance via gradient ascent.

    Maximises Jaccard displacement of victim's top-k by perturbing
    adversary's ratings. Uses a differentiable surrogate for displacement.

    Args:
        recommender: A Recommender with get_scores_differentiable().
        victim_id: User whose recommendations we monitor.
        adversary_id: User whose ratings we perturb.
        rating_matrix: Original rating matrix.
        k: Top-k list size.
        epsilon: Budget — max ‖δ_a‖₁ (number of rating changes).
        lr: Learning rate.
        n_steps: Gradient steps.

    Returns:
        dict with displacement (Jaccard distance), perturbation, original/new top-k.
    """
    n, m = rating_matrix.shape
    R = torch.tensor(rating_matrix, dtype=torch.float32)

    # Original top-k for victim
    orig_scores = recommender.get_scores(victim_id, rating_matrix)
    rated_mask = rating_matrix[victim_id] > 0
    orig_scores_masked = orig_scores.copy()
    orig_scores_masked[rated_mask] = -np.inf
    orig_topk = set(np.argsort(orig_scores_masked)[::-1][:k].tolist())

    # Perturbation to adversary's row
    delta_a = torch.zeros(m, requires_grad=True)
    optimizer = torch.optim.Adam([delta_a], lr=lr)

    best_displacement = 0.0
    best_delta = None
    best_new_topk = orig_topk

    for step in range(n_steps):
        optimizer.zero_grad()

        # Perturb adversary's row
        R_pert = R.clone()
        R_pert[adversary_id] = torch.clamp(R[adversary_id] + delta_a, 0.0, 5.0)

        # Get victim's scores under perturbed matrix
        # We need to retrain/recompute — for MF, use differentiable path
        victim_scores = recommender.get_scores_differentiable(victim_id, R_pert[victim_id])

        # Differentiable surrogate: maximise negative correlation with original ranking
        orig_scores_t = torch.tensor(orig_scores, dtype=torch.float32)
        # Push scores of original top-k items down, others up
        orig_topk_list = list(orig_topk)
        topk_mask = torch.zeros(m)
        topk_mask[orig_topk_list] = 1.0

        # Loss: minimise scores of original top-k items (maximise displacement)
        loss = (victim_scores * topk_mask).sum() - (victim_scores * (1 - topk_mask)).mean()

        loss.backward()
        optimizer.step()

        # Project onto L1 ball
        with torch.no_grad():
            if delta_a.abs().sum() > epsilon:
                delta_a.data = _project_l1(delta_a.data, epsilon)

        # Evaluate actual Jaccard displacement
        with torch.no_grad():
            R_eval = rating_matrix.copy()
            R_eval[adversary_id] = np.clip(
                rating_matrix[adversary_id] + delta_a.numpy(), 0.0, 5.0
            )
            new_scores = recommender.get_scores(victim_id, R_eval)
            new_scores_masked = new_scores.copy()
            new_scores_masked[rating_matrix[victim_id] > 0] = -np.inf
            new_topk = set(np.argsort(new_scores_masked)[::-1][:k].tolist())
            displacement = _jaccard_distance(orig_topk, new_topk)

            if displacement > best_displacement:
                best_displacement = displacement
                best_delta = delta_a.detach().clone()
                best_new_topk = new_topk

    if best_delta is None:
        best_delta = delta_a.detach()

    return {
        "displacement": best_displacement,
        "perturbation": best_delta.numpy(),
        "original_topk": orig_topk,
        "new_topk": best_new_topk,
        "epsilon": epsilon,
    }


def manipulation_resistance_blackbox(
    recommender,
    victim_id: int,
    adversary_id: int,
    rating_matrix: np.ndarray,
    k: int = 10,
    epsilon: int = 10,
    rating_values: list[float] | None = None,
    n_random_restarts: int = 3,
) -> dict:
    """Black-box manipulation resistance via greedy hill-climbing.

    Greedily adds/changes adversary ratings to maximise Jaccard displacement
    of victim's top-k.

    Args:
        recommender: A Recommender with get_scores().
        victim_id: User whose recommendations we monitor.
        adversary_id: User whose ratings we perturb.
        rating_matrix: Original rating matrix.
        k: Top-k list size.
        epsilon: Budget — max number of rating changes.
        rating_values: Discrete rating values to try.
        n_random_restarts: Number of random starting points to try.

    Returns:
        dict with displacement, original/new top-k.
    """
    if rating_values is None:
        rating_values = [1.0, 2.0, 3.0, 4.0, 5.0]

    m = rating_matrix.shape[1]

    # Original top-k
    orig_scores = recommender.get_scores(victim_id, rating_matrix)
    rated_mask = rating_matrix[victim_id] > 0
    orig_scores[rated_mask] = -np.inf
    orig_topk = set(np.argsort(orig_scores)[::-1][:k].tolist())

    best_displacement = 0.0
    best_new_topk = orig_topk

    for restart in range(n_random_restarts):
        R = rating_matrix.copy()
        changes = []

        for step in range(epsilon):
            # Candidate items for adversary to rate
            candidates = np.where(R[adversary_id] == 0)[0]
            if len(candidates) > 200:
                candidates = np.random.choice(candidates, 200, replace=False)

            best_step_disp = 0.0
            best_item = -1
            best_val = 0.0

            for item in candidates:
                for val in rating_values:
                    R_try = R.copy()
                    R_try[adversary_id, item] = val
                    scores = recommender.get_scores(victim_id, R_try)
                    scores_masked = scores.copy()
                    scores_masked[rating_matrix[victim_id] > 0] = -np.inf
                    new_topk = set(np.argsort(scores_masked)[::-1][:k].tolist())
                    disp = _jaccard_distance(orig_topk, new_topk)
                    if disp > best_step_disp:
                        best_step_disp = disp
                        best_item = item
                        best_val = val

            if best_item == -1:
                break

            R[adversary_id, best_item] = best_val
            changes.append((best_item, best_val))

            if best_step_disp > best_displacement:
                best_displacement = best_step_disp
                scores = recommender.get_scores(victim_id, R)
                scores_masked = scores.copy()
                scores_masked[rating_matrix[victim_id] > 0] = -np.inf
                best_new_topk = set(np.argsort(scores_masked)[::-1][:k].tolist())

    return {
        "displacement": best_displacement,
        "original_topk": orig_topk,
        "new_topk": best_new_topk,
        "epsilon": epsilon,
    }


def _project_l1(x: torch.Tensor, radius: float) -> torch.Tensor:
    """Project x onto the L1 ball of given radius."""
    if x.abs().sum() <= radius:
        return x
    # Soft thresholding
    u = x.abs().sort(descending=True).values
    cumsum = torch.cumsum(u, dim=0)
    rho = torch.where(u > (cumsum - radius) / torch.arange(1, len(u) + 1, dtype=x.dtype))[0]
    if len(rho) == 0:
        return torch.zeros_like(x)
    rho = rho[-1]
    theta = max(0, (cumsum[rho] - radius) / (rho + 1))
    return torch.sign(x) * torch.clamp(x.abs() - theta, min=0)


def manipulation_resistance(
    recommender,
    victim_id: int,
    adversary_id: int,
    rating_matrix: np.ndarray,
    k: int = 10,
    epsilon: int = 10,
    mode: str = "whitebox",
    **kwargs,
) -> dict:
    """Compute manipulation resistance M(i, a, ε).

    Args:
        mode: "whitebox" or "blackbox".
    """
    if mode == "whitebox":
        return manipulation_resistance_whitebox(
            recommender, victim_id, adversary_id, rating_matrix, k, epsilon, **kwargs
        )
    elif mode == "blackbox":
        return manipulation_resistance_blackbox(
            recommender, victim_id, adversary_id, rating_matrix, k, epsilon, **kwargs
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'whitebox' or 'blackbox'.")

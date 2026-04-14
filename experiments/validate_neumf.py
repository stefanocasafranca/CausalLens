"""Phase 2 validation: NeuMF on MovieLens-1M, all metrics, 10 users.

NeuMF has cross-user coupling via shared item embeddings — retraining
after adversary perturbation propagates effects to victim's scores.
This should produce non-zero manipulation resistance (unlike frozen-V MF).

Uses lightweight random-search variants for reachability and manipulation
because each NeuMF get_scores call triggers a finetune/retrain.
"""

import sys
import numpy as np
import time
np.random.seed(42)

# Unbuffered output
print = lambda *args, **kwargs: (sys.stdout.write(" ".join(str(a) for a in args) + kwargs.get("end", "\n")), sys.stdout.flush())

from causallens.data.movielens import load_movielens_1m
from causallens.models.neumf import NeuMF
from causallens.metrics.observational import (
    intra_list_diversity,
    recommendation_volatility,
    catalog_coverage,
)

t0 = time.time()


def _jaccard_distance(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def reachability_random_search(
    recommender, user_id, target_item, rating_matrix,
    k=10, max_budget=10, n_trials=30,
):
    """Random search reachability for neural models.

    Instead of greedy hill-climbing (200 candidates × values per step),
    randomly perturb the user's ratings and check if the target enters top-k.
    Each trial tries increasing budgets: 1, 2, ..., max_budget.
    """
    m = rating_matrix.shape[1]
    unrated = np.where(rating_matrix[user_id] == 0)[0]
    unrated = unrated[unrated != target_item]
    if len(unrated) == 0:
        return {"cost": float("inf"), "success": False, "rank": m}

    # Get current rank
    scores = recommender.get_scores(user_id, rating_matrix)
    scores_masked = scores.copy()
    scores_masked[rating_matrix[user_id] > 0] = -np.inf
    orig_rank = int((scores_masked > scores_masked[target_item]).sum()) + 1

    if orig_rank <= k:
        return {"cost": 0.0, "success": True, "rank": orig_rank}

    best_cost = float("inf")
    best_rank = orig_rank
    rating_values = [1.0, 2.0, 3.0, 4.0, 5.0]

    for trial in range(n_trials):
        R = rating_matrix.copy()
        budget = np.random.randint(1, max_budget + 1)
        items = np.random.choice(unrated, min(budget, len(unrated)), replace=False)
        for item in items:
            R[user_id, item] = np.random.choice(rating_values)

        scores = recommender.get_scores(user_id, R)
        scores_masked = scores.copy()
        scores_masked[R[user_id] > 0] = -np.inf
        rank = int((scores_masked > scores_masked[target_item]).sum()) + 1

        if rank <= k and len(items) < best_cost:
            best_cost = len(items)
            best_rank = rank

    return {
        "cost": best_cost,
        "success": best_cost < float("inf"),
        "rank": best_rank,
    }


def manipulation_random_search(
    recommender, victim_id, adversary_id, rating_matrix,
    k=10, epsilon=5, n_trials=20,
):
    """Random search manipulation for retrain-based models."""
    orig_scores = recommender.get_scores(victim_id, rating_matrix)
    rated_mask = rating_matrix[victim_id] > 0
    orig_scores[rated_mask] = -np.inf
    orig_topk = set(np.argsort(orig_scores)[::-1][:k].tolist())

    best_disp = 0.0
    best_topk = orig_topk
    rating_values = [1.0, 2.0, 3.0, 4.0, 5.0]

    unrated = np.where(rating_matrix[adversary_id] == 0)[0]
    if len(unrated) == 0:
        return {"displacement": 0.0, "original_topk": orig_topk, "new_topk": orig_topk}

    for trial in range(n_trials):
        R = rating_matrix.copy()
        items = np.random.choice(unrated, min(epsilon, len(unrated)), replace=False)
        for item in items:
            R[adversary_id, item] = np.random.choice(rating_values)

        scores = recommender.get_scores(victim_id, R)
        scores[rating_matrix[victim_id] > 0] = -np.inf
        new_topk = set(np.argsort(scores)[::-1][:k].tolist())
        disp = _jaccard_distance(orig_topk, new_topk)
        if disp > best_disp:
            best_disp = disp
            best_topk = new_topk

    return {"displacement": best_disp, "original_topk": orig_topk, "new_topk": best_topk}


def self_influence_random_search(
    recommender, user_id, rating_matrix, k=10, epsilon=5, n_trials=20,
):
    """Random search self-influence."""
    orig_topk = set(
        recommender.get_recommendations(user_id, rating_matrix, k).tolist()
    )
    rated = np.where(rating_matrix[user_id] > 0)[0]
    if len(rated) == 0:
        return 0.0

    rating_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    best_disp = 0.0

    for _ in range(n_trials):
        R = rating_matrix.copy()
        items = np.random.choice(rated, min(epsilon, len(rated)), replace=False)
        for item in items:
            R[user_id, item] = np.random.choice(rating_values)

        scores = recommender.get_scores(user_id, R)
        scores[R[user_id] > 0] = -np.inf
        new_topk = set(np.argsort(scores)[::-1][:k].tolist())
        disp = _jaccard_distance(orig_topk, new_topk)
        if disp > best_disp:
            best_disp = disp

    return best_disp


# ── Load data ──────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1: Loading MovieLens-1M")
print("=" * 70)
data = load_movielens_1m()
R = data["rating_matrix"]
n_users, n_items = R.shape

# ── Train NeuMF ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Training NeuMF")
print("=" * 70)
neumf = NeuMF(
    n_users, n_items,
    gmf_dim=32, mlp_dims=(64, 32, 16),
    n_epochs=10, batch_size=1024,
    retrain_steps=50, finetune_steps=20,
)
neumf.fit(R)

# ── Setup ──────────────────────────────────────────────────────────────
k = 10
epsilon = 5

# Sample 10 active users
active = np.where((R > 0).sum(axis=1) >= 20)[0]
user_ids = np.random.choice(active, 10, replace=False).tolist()
print(f"\nSampled users: {user_ids}")

# ── Reachability (random search, 2 targets each) ──────────────────────
print("\n" + "=" * 70)
print("STEP 3: Reachability (random search, 2 targets/user, 30 trials)")
print("=" * 70)

reach_results = []
for uid in user_ids:
    current_topk = set(neumf.get_recommendations(uid, R, k).tolist())
    unrated = np.where(R[uid] == 0)[0]
    available = [i for i in unrated if i not in current_topk]
    targets = np.random.choice(available, min(2, len(available)), replace=False)
    t_start = time.time()
    for t in targets:
        r = reachability_random_search(neumf, uid, int(t), R, k=k,
                                       max_budget=10, n_trials=30)
        r["user_id"] = uid
        r["target_item"] = t
        reach_results.append(r)
    successes = sum(1 for r in reach_results if r["user_id"] == uid and r["success"])
    print(f"  User {uid}: {successes}/{len(targets)} reached ({time.time()-t_start:.1f}s)")

costs = [r["cost"] for r in reach_results if r["success"]]
success_rate = sum(1 for r in reach_results if r["success"]) / max(len(reach_results), 1)
print(f"  Success rate: {success_rate:.2%}")
print(f"  Mean cost:    {np.mean(costs):.2f}" if costs else "  Mean cost:    N/A")

# ── Manipulation Resistance (random search, 1 adversary each) ──────────
print("\n" + "=" * 70)
print("STEP 4: Manipulation Resistance (random search, 1 adversary, 15 trials)")
print("=" * 70)

manip_results = []
for uid in user_ids:
    others = [u for u in range(n_users) if u != uid]
    aid = int(np.random.choice(others, 1)[0])
    t_start = time.time()
    r = manipulation_random_search(neumf, uid, aid, R, k=k, epsilon=epsilon,
                                   n_trials=15)
    r["victim_id"] = uid
    r["adversary_id"] = aid
    manip_results.append(r)
    print(f"  User {uid} vs {aid}: displacement={r['displacement']:.4f} "
          f"({time.time()-t_start:.1f}s)")

disps = [r["displacement"] for r in manip_results]
print(f"  Mean displacement: {np.mean(disps):.4f}")
print(f"  Max displacement:  {np.max(disps):.4f}")

# ── AAI ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: Autonomy Asymmetry Index (AAI)")
print("=" * 70)

aai_results = []
for i, uid in enumerate(user_ids):
    t_start = time.time()
    # Self-influence via random search
    s = self_influence_random_search(neumf, uid, R, k=k, epsilon=epsilon,
                                     n_trials=15)
    # Adversary influence from manipulation results
    e = manip_results[i]["displacement"]

    if s < 1e-10:
        aai = float("inf") if e > 0 else 1.0
    else:
        aai = e / s

    label = "healthy" if aai < 1.0 else ("borderline" if aai == 1.0 else "problematic")
    aai_results.append({
        "aai": aai, "self_influence": s, "adversary_influence": e, "label": label,
    })
    print(f"  User {uid}: S={s:.4f}, E={e:.4f}, AAI={aai:.4f} ({label}) "
          f"({time.time()-t_start:.1f}s)")

aai_vals = [r["aai"] for r in aai_results if r["aai"] != float("inf")]
n_prob = sum(1 for r in aai_results if r["aai"] > 1.0)
print(f"  Mean AAI:      {np.mean(aai_vals):.4f}" if aai_vals else "  Mean AAI: N/A")
print(f"  # Problematic: {n_prob}/{len(user_ids)}")

# ── Observational Baselines ────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: Observational Baselines")
print("=" * 70)

diversities = []
volatilities = []
for uid in user_ids:
    diversities.append(intra_list_diversity(neumf, uid, R, k))
    volatilities.append(recommendation_volatility(neumf, uid, R, k))

coverage = catalog_coverage(neumf, user_ids, R, k)
print(f"  Mean diversity:   {np.mean(diversities):.4f}")
print(f"  Mean volatility:  {np.mean(volatilities):.4f}")
print(f"  Catalog coverage: {coverage:.4f}")

# ── Summary Table ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS TABLE — NeuMF Per-User Summary")
print("=" * 70)

header = (f"{'User':>6} | {'Reach':>8} | {'Manip':>8} | {'SelfInf':>8} | "
          f"{'AdvInf':>8} | {'AAI':>8} | {'Label':>12} | {'Diversity':>9} | {'Volatility':>10}")
print(header)
print("-" * len(header))

for i, uid in enumerate(user_ids):
    user_reach = [r for r in reach_results if r["user_id"] == uid]
    avg_cost = np.mean([r["cost"] for r in user_reach]) if user_reach else float("nan")

    aai_r = aai_results[i]
    aai_val = aai_r["aai"]
    aai_str = f"{aai_val:.4f}" if aai_val != float("inf") else "inf"

    print(f"{uid:>6} | {avg_cost:>8.2f} | {aai_r['adversary_influence']:>8.4f} | "
          f"{aai_r['self_influence']:>8.4f} | {aai_r['adversary_influence']:>8.4f} | "
          f"{aai_str:>8} | {aai_r['label']:>12} | {diversities[i]:>9.4f} | {volatilities[i]:>10.4f}")

elapsed = time.time() - t0
print(f"\n{'=' * 70}")
print(f"NeuMF VALIDATION COMPLETE — all metrics in {elapsed:.1f}s")
print(f"{'=' * 70}")

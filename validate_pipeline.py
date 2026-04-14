"""Quick validation: train MF on MovieLens-1M, run all metrics on 10 users."""

import numpy as np
import time
np.random.seed(42)

from causallens.data.movielens import load_movielens_1m
from causallens.models.mf import MatrixFactorization
from causallens.core import CausalLens
from causallens.metrics.reachability import reachability_cost
from causallens.metrics.manipulation import manipulation_resistance
from causallens.metrics.aai import autonomy_asymmetry_index
from causallens.metrics.observational import (
    intra_list_diversity,
    recommendation_volatility,
    catalog_coverage,
)

t0 = time.time()

# ── Load data ──────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1: Loading MovieLens-1M")
print("=" * 70)
data = load_movielens_1m()
R = data["rating_matrix"]
n_users, n_items = R.shape

# ── Train MF ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Training Matrix Factorization")
print("=" * 70)
mf = MatrixFactorization(n_users, n_items, n_factors=32, n_epochs=10, lr=1e-3)
mf.fit(R)

# ── Set up ─────────────────────────────────────────────────────────────
k = 10
epsilon = 5
lens = CausalLens(mf, R, k=k)

# Sample 10 users
user_ids = lens._sample_users(n=10)
print(f"\nSampled users: {user_ids}")

# ── Reachability (whitebox, 3 targets each) ────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Reachability (whitebox, 3 targets/user)")
print("=" * 70)

reach_results = []
for uid in user_ids:
    current_topk = set(mf.get_recommendations(uid, R, k).tolist())
    unrated = np.where(R[uid] == 0)[0]
    available = [i for i in unrated if i not in current_topk]
    targets = np.random.choice(available, min(3, len(available)), replace=False)
    for t in targets:
        r = reachability_cost(mf, uid, int(t), R, k=k, max_budget=50,
                              mode="whitebox", n_steps=80, lr=1.0)
        r["user_id"] = uid
        r["target_item"] = t
        reach_results.append(r)
    successes = sum(1 for r in reach_results if r["user_id"] == uid and r["success"])
    print(f"  User {uid}: {successes}/{len(targets)} reached, "
          f"ranks={[r['rank'] for r in reach_results if r['user_id']==uid]}")

costs = [r["cost"] for r in reach_results if r["success"]]
success_rate = sum(1 for r in reach_results if r["success"]) / max(len(reach_results), 1)
print(f"  Success rate: {success_rate:.2%}")
print(f"  Mean cost:    {np.mean(costs):.2f}" if costs else "  Mean cost:    N/A")

# ── Manipulation Resistance (whitebox, 2 adversaries each) ────────────
print("\n" + "=" * 70)
print("STEP 4: Manipulation Resistance (whitebox, 2 adversaries/victim)")
print("=" * 70)

manip_results = []
for uid in user_ids:
    others = [u for u in range(n_users) if u != uid]
    advs = np.random.choice(others, 2, replace=False)
    for aid in advs:
        r = manipulation_resistance(mf, uid, int(aid), R, k=k, epsilon=epsilon,
                                    mode="whitebox", n_steps=40)
        r["victim_id"] = uid
        r["adversary_id"] = aid
        manip_results.append(r)
    print(f"  User {uid}: 2 adversaries done")

disps = [r["displacement"] for r in manip_results]
print(f"  Mean displacement: {np.mean(disps):.4f}")
print(f"  Max displacement:  {np.max(disps):.4f}")

# ── AAI ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: Autonomy Asymmetry Index (AAI)")
print("=" * 70)

aai_results = []
for uid in user_ids:
    others = [u for u in range(n_users) if u != uid]
    advs = np.random.choice(others, 2, replace=False).tolist()
    r = autonomy_asymmetry_index(mf, uid, R, k=k, epsilon=epsilon,
                                 adversary_ids=advs, mode="whitebox",
                                 self_trials=2)
    aai_results.append(r)
    print(f"  User {uid}: AAI={r['aai']:.4f} ({r['label']})")

aai_vals = [r["aai"] for r in aai_results if r["aai"] != float("inf")]
n_prob = sum(1 for r in aai_results if r["aai"] > 1.0)
print(f"  Mean AAI:      {np.mean(aai_vals):.4f}" if aai_vals else "  Mean AAI: N/A")
print(f"  # Problematic: {n_prob}/{len(user_ids)}")

# ── Observational Baselines ──────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: Observational Baselines")
print("=" * 70)

diversities = []
volatilities = []
for uid in user_ids:
    diversities.append(intra_list_diversity(mf, uid, R, k))
    volatilities.append(recommendation_volatility(mf, uid, R, k))

coverage = catalog_coverage(mf, user_ids, R, k)
print(f"  Mean diversity:   {np.mean(diversities):.4f}")
print(f"  Mean volatility:  {np.mean(volatilities):.4f}")
print(f"  Catalog coverage: {coverage:.4f}")

# ── Summary Table ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS TABLE — Per-User Summary")
print("=" * 70)

header = f"{'User':>6} | {'Reach':>8} | {'Manip':>8} | {'SelfInf':>8} | {'AdvInf':>8} | {'AAI':>8} | {'Label':>12} | {'Diversity':>9} | {'Volatility':>10}"
print(header)
print("-" * len(header))

for i, uid in enumerate(user_ids):
    user_reach = [r for r in reach_results if r["user_id"] == uid]
    avg_cost = np.mean([r["cost"] for r in user_reach]) if user_reach else float("nan")

    user_manip = [r for r in manip_results if r["victim_id"] == uid]
    avg_disp = np.mean([r["displacement"] for r in user_manip]) if user_manip else float("nan")

    aai_r = aai_results[i]
    aai_val = aai_r["aai"]
    aai_str = f"{aai_val:.4f}" if aai_val != float("inf") else "inf"
    s_inf = aai_r["self_influence"]
    a_inf = aai_r["adversary_influence"]
    label = aai_r["label"]

    div = diversities[i]
    vol = volatilities[i]

    print(f"{uid:>6} | {avg_cost:>8.2f} | {avg_disp:>8.4f} | {s_inf:>8.4f} | {a_inf:>8.4f} | {aai_str:>8} | {label:>12} | {div:>9.4f} | {vol:>10.4f}")

elapsed = time.time() - t0
print(f"\n{'=' * 70}")
print(f"VALIDATION COMPLETE — all metrics computed in {elapsed:.1f}s")
print(f"{'=' * 70}")

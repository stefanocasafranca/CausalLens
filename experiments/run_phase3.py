"""Phase 3: Full experiment pipeline — 4 models × 2 datasets × 200 users.

Runs MF, NeuMF, LightGCN, and SASRec on MovieLens-1M and Amazon Digital
Music with all metrics: reachability, manipulation resistance, AAI,
observational baselines, ODR.  Saves per-user results to CSV.

Usage:
    PYTHONPATH=. python experiments/run_phase3.py
"""

import sys
import os
import csv
import time
import signal
import numpy as np
import torch
from tqdm import tqdm, trange

np.random.seed(42)

# Unbuffered output
_builtin_print = print
def print(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

from causallens.data.movielens import load_movielens_1m
from causallens.data.amazon import load_amazon_digital_music
from causallens.models.mf import MatrixFactorization
from causallens.models.neumf import NeuMF
from causallens.models.lightgcn import LightGCN
from causallens.models.sasrec import SASRec
from causallens.metrics.observational import (
    intra_list_diversity,
    recommendation_volatility,
    catalog_coverage,
)

# ── Config ────────────────────────────────────────────────────────────
N_USERS = 200
K = 10
EPSILON = 5
N_REACHABILITY_TARGETS = 5
N_ADVERSARIES = 3
REACHABILITY_MAX_BUDGET = 20
REACHABILITY_TRIALS = 50       # random search trials (same solver for all models)
MANIPULATION_TRIALS = 10       # default manipulation trials
SELF_INFLUENCE_TRIALS = 15     # default self-influence trials
# Per-model trial overrides (heavier models get fewer trials to keep runtime bounded)
MODEL_MANIPULATION_TRIALS = {"MF": 10, "NeuMF": 8, "LightGCN": 2, "SASRec": 2}
MODEL_SELF_INFLUENCE_TRIALS = {"MF": 15, "NeuMF": 10, "LightGCN": 10, "SASRec": 10}
MODEL_REACHABILITY_TRIALS = {"MF": 50, "NeuMF": 50, "LightGCN": 15, "SASRec": 15}
USER_TIMEOUT = 180           # seconds per user (safety net)
CSV_PATH = "results/phase3_results.csv"
MAX_TOTAL_HOURS = 48.0       # large budget for 4 models × 2 datasets unattended

# ── Timeout helper ────────────────────────────────────────────────────

class TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutError("Timed out")


# ── Helper functions ──────────────────────────────────────────────────

def _jaccard_distance(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def reachability_random_search(
    recommender, user_id, target_item, rating_matrix,
    k=10, max_budget=50, n_trials=50,
):
    """Random search reachability (same solver for all models)."""
    m = rating_matrix.shape[1]
    unrated = np.where(rating_matrix[user_id] == 0)[0]
    unrated = unrated[unrated != target_item]
    if len(unrated) == 0:
        return {"cost": float("inf"), "success": False, "rank": m}

    scores = recommender.get_scores(user_id, rating_matrix)
    scores_masked = scores.copy()
    scores_masked[rating_matrix[user_id] > 0] = -np.inf
    orig_rank = int((scores_masked > scores_masked[target_item]).sum()) + 1

    if orig_rank <= k:
        return {"cost": 0.0, "success": True, "rank": orig_rank}

    best_cost = float("inf")
    best_rank = orig_rank
    rating_values = [1.0, 2.0, 3.0, 4.0, 5.0]

    for _ in range(n_trials):
        budget = np.random.randint(1, max_budget + 1)
        items = np.random.choice(unrated, min(budget, len(unrated)), replace=False)
        old_vals = rating_matrix[user_id, items].copy()
        try:
            for item in items:
                rating_matrix[user_id, item] = np.random.choice(rating_values)

            scores = recommender.get_scores(user_id, rating_matrix)
            scores_masked = scores.copy()
            scores_masked[rating_matrix[user_id] > 0] = -np.inf
            rank = int((scores_masked > scores_masked[target_item]).sum()) + 1

            if rank <= k and len(items) < best_cost:
                best_cost = len(items)
                best_rank = rank
        finally:
            rating_matrix[user_id, items] = old_vals

    return {
        "cost": best_cost,
        "success": best_cost < float("inf"),
        "rank": best_rank,
    }


def manipulation_random_search(
    recommender, victim_id, adversary_id, rating_matrix,
    k=10, epsilon=5, n_trials=15,
):
    """Random search manipulation for retrain-based models."""
    orig_scores = recommender.get_scores(victim_id, rating_matrix)
    rated_mask = rating_matrix[victim_id] > 0
    orig_scores[rated_mask] = -np.inf
    orig_topk = set(np.argsort(orig_scores)[::-1][:k].tolist())

    best_disp = 0.0
    rating_values = [1.0, 2.0, 3.0, 4.0, 5.0]

    unrated = np.where(rating_matrix[adversary_id] == 0)[0]
    if len(unrated) == 0:
        return {"displacement": 0.0}

    for _ in range(n_trials):
        items = np.random.choice(unrated, min(epsilon, len(unrated)), replace=False)
        old_vals = rating_matrix[adversary_id, items].copy()
        try:
            for item in items:
                rating_matrix[adversary_id, item] = np.random.choice(rating_values)

            scores = recommender.get_scores(victim_id, rating_matrix)
            scores[rating_matrix[victim_id] > 0] = -np.inf
            new_topk = set(np.argsort(scores)[::-1][:k].tolist())

            disp = _jaccard_distance(orig_topk, new_topk)
            if disp > best_disp:
                best_disp = disp
        finally:
            rating_matrix[adversary_id, items] = old_vals

    return {"displacement": best_disp}


def self_influence_random_search(
    recommender, user_id, rating_matrix, k=10, epsilon=5, n_trials=20,
):
    """Random search self-influence.

    Mixes three action types: modify existing ratings, add new ratings
    (rate unrated items), and remove existing ratings (set to 0).  This
    ensures graph-based models (LightGCN, SASRec) whose scoring depends
    on edge presence—not rating values—see structural changes.
    """
    orig_topk = set(
        recommender.get_recommendations(user_id, rating_matrix, k).tolist()
    )
    rated = np.where(rating_matrix[user_id] > 0)[0]
    unrated = np.where(rating_matrix[user_id] == 0)[0]
    if len(rated) == 0 and len(unrated) == 0:
        return 0.0

    rating_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    best_disp = 0.0

    for _ in range(n_trials):
        # Decide split: how many to modify vs add vs remove
        n_modify = np.random.randint(0, min(epsilon, len(rated)) + 1) if len(rated) > 0 else 0
        remaining = epsilon - n_modify
        n_add = np.random.randint(0, min(remaining, len(unrated)) + 1) if len(unrated) > 0 else 0
        remaining2 = remaining - n_add
        n_remove = min(remaining2, len(rated) - n_modify) if len(rated) > n_modify else 0

        modify_items = np.random.choice(rated, n_modify, replace=False) if n_modify > 0 else np.array([], dtype=np.int64)
        add_items = np.random.choice(unrated, n_add, replace=False) if n_add > 0 else np.array([], dtype=np.int64)
        # For removals, pick from rated items not already being modified
        remaining_rated = np.setdiff1d(rated, modify_items)
        remove_items = np.random.choice(remaining_rated, n_remove, replace=False) if n_remove > 0 and len(remaining_rated) > 0 else np.array([], dtype=np.int64)

        all_items = np.concatenate([modify_items, add_items, remove_items]).astype(np.int64)
        if len(all_items) == 0:
            continue
        old_vals = rating_matrix[user_id, all_items].copy()
        try:
            for item in modify_items:
                rating_matrix[user_id, item] = np.random.choice(rating_values)
            for item in add_items:
                rating_matrix[user_id, item] = np.random.choice(rating_values)
            for item in remove_items:
                rating_matrix[user_id, item] = 0.0

            scores = recommender.get_scores(user_id, rating_matrix)
            scores[rating_matrix[user_id] > 0] = -np.inf
            new_topk = set(np.argsort(scores)[::-1][:k].tolist())
        finally:
            rating_matrix[user_id, all_items] = old_vals
        disp = _jaccard_distance(orig_topk, new_topk)
        if disp > best_disp:
            best_disp = disp

    return best_disp


# ── Per-user metric computation ───────────────────────────────────────

def _default_user_result(uid, model_name):
    """Return default result when a user times out."""
    return {
        "user_id": uid, "model": model_name,
        "reachability_mean": float("inf"), "reachability_success_rate": 0.0,
        "manipulation_resistance_mean": 0.0, "self_influence": 0.0,
        "external_influence": 0.0, "aai": 1.0, "aai_label": "borderline",
        "diversity": 0.0, "volatility": 0.0,
    }


def compute_user_metrics(
    model, model_name, uid, R, all_user_ids, n_users,
):
    """Compute all per-user metrics. Returns a dict."""
    # For non-MF models, wrap entire computation in a timeout
    if model_name != "MF":
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(USER_TIMEOUT)
        try:
            result = _compute_user_metrics_inner(model, model_name, uid, R, all_user_ids, n_users)
            signal.alarm(0)
            return result
        except TimeoutError:
            signal.alarm(0)
            return _default_user_result(uid, model_name)
    else:
        return _compute_user_metrics_inner(model, model_name, uid, R, all_user_ids, n_users)


def _compute_user_metrics_inner(
    model, model_name, uid, R, all_user_ids, n_users,
):
    """Inner function for per-user metric computation."""
    result = {"user_id": uid, "model": model_name}

    # --- Reachability (5 targets) ---
    current_topk = set(model.get_recommendations(uid, R, K).tolist())
    unrated = np.where(R[uid] == 0)[0]
    available = [i for i in unrated if i not in current_topk]
    if len(available) < N_REACHABILITY_TARGETS:
        targets = available
    else:
        targets = np.random.choice(available, N_REACHABILITY_TARGETS, replace=False)

    costs = []
    successes = 0
    consecutive_failures = 0
    for t in targets:
        r = reachability_random_search(
            model, uid, int(t), R, k=K,
            max_budget=REACHABILITY_MAX_BUDGET,
            n_trials=MODEL_REACHABILITY_TRIALS.get(model_name, REACHABILITY_TRIALS),
        )
        if r["success"]:
            successes += 1
            costs.append(r["cost"])
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            # Early termination: if 3+ consecutive targets unreachable, skip rest
            if consecutive_failures >= 3 and successes == 0:
                break

    n_targets = max(len(targets), 1)
    result["reachability_mean"] = float(np.mean(costs)) if costs else float("inf")
    result["reachability_success_rate"] = successes / n_targets

    # --- Manipulation Resistance (3 adversaries) ---
    other_users = [u for u in all_user_ids if u != uid]
    if len(other_users) < N_ADVERSARIES:
        adversaries = other_users
    else:
        adversaries = np.random.choice(other_users, N_ADVERSARIES, replace=False).tolist()

    n_manip_trials = MODEL_MANIPULATION_TRIALS.get(model_name, MANIPULATION_TRIALS)
    displacements = []
    for aid in adversaries:
        mr = manipulation_random_search(
            model, uid, aid, R, k=K, epsilon=EPSILON,
            n_trials=n_manip_trials,
        )
        displacements.append(mr["displacement"])

    result["manipulation_resistance_mean"] = float(np.mean(displacements))
    result["external_influence"] = float(np.max(displacements))

    # --- Self-influence ---
    n_self_trials = MODEL_SELF_INFLUENCE_TRIALS.get(model_name, SELF_INFLUENCE_TRIALS)
    s = self_influence_random_search(
        model, uid, R, k=K, epsilon=EPSILON,
        n_trials=n_self_trials,
    )
    result["self_influence"] = s

    # --- AAI ---
    e = result["external_influence"]
    if s < 1e-10:
        aai = float("inf") if e > 0 else 1.0
    else:
        aai = e / s
    result["aai"] = aai
    result["aai_label"] = "healthy" if aai < 1.0 else ("borderline" if aai == 1.0 else "problematic")

    # --- Observational ---
    result["diversity"] = intra_list_diversity(model, uid, R, K)
    result["volatility"] = recommendation_volatility(model, uid, R, K)
    # coverage is population-level, filled in later

    return result


# ── ODR computation from results ──────────────────────────────────────

def compute_odr(results):
    """Derive ODR from already-computed per-user results.

    A user is 'observationally autonomous' if diversity > median AND
    volatility > median. A user is 'causally trapped' if all reachability
    targets failed (success_rate == 0). ODR = fraction of observationally
    autonomous users who are causally trapped.
    """
    divs = [r["diversity"] for r in results]
    vols = [r["volatility"] for r in results]
    med_div = float(np.median(divs))
    med_vol = float(np.median(vols))

    obs_autonomous = []
    for r in results:
        if r["diversity"] > med_div and r["volatility"] > med_vol:
            obs_autonomous.append(r)

    if len(obs_autonomous) == 0:
        return {"odr": 0.0, "n_autonomous": 0, "n_deceptive": 0,
                "med_diversity": med_div, "med_volatility": med_vol}

    n_trapped = sum(1 for r in obs_autonomous if r["reachability_success_rate"] == 0.0)
    odr = n_trapped / len(obs_autonomous)

    # Mark deceptive users
    for r in results:
        is_autonomous = r["diversity"] > med_div and r["volatility"] > med_vol
        is_trapped = r["reachability_success_rate"] == 0.0
        r["deceptive"] = 1 if (is_autonomous and is_trapped) else 0

    return {"odr": odr, "n_autonomous": len(obs_autonomous), "n_deceptive": n_trapped,
            "med_diversity": med_div, "med_volatility": med_vol}


# ── Main experiment loop ──────────────────────────────────────────────

def _rewrite_csv_final(csv_path, fieldnames, all_combo_results):
    """Rewrite the CSV with proper coverage and deceptive columns."""
    # Build lookup: (model, dataset) -> {coverage, odr_info}
    combo_info = {}
    for s in all_combo_results:
        combo_info[(s["model"], s["dataset"])] = s

    # Read all rows
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["model"], row["dataset"])
            if key in combo_info:
                row["coverage"] = combo_info[key]["coverage"]
            rows.append(row)

    # Now recompute deceptive per combo
    from collections import defaultdict
    combo_rows = defaultdict(list)
    for row in rows:
        combo_rows[(row["model"], row["dataset"])].append(row)

    for key, crow in combo_rows.items():
        divs = [float(r["diversity"]) for r in crow]
        vols = [float(r["volatility"]) for r in crow]
        med_div = float(np.median(divs))
        med_vol = float(np.median(vols))
        for r in crow:
            is_auto = float(r["diversity"]) > med_div and float(r["volatility"]) > med_vol
            sr = float(r["reachability_success_rate"])
            r["deceptive"] = 1 if (is_auto and sr == 0.0) else 0

    # Rewrite
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _load_completed(csv_path):
    """Load already-completed (model, dataset, user_id) from CSV for resume."""
    done = set()
    if not os.path.exists(csv_path):
        return done, []
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["model"], row["dataset"], int(row["user_id"]))
            done.add(key)
            rows.append(row)
    return done, rows


def run_experiment():
    os.makedirs("results", exist_ok=True)

    # CSV setup
    fieldnames = [
        "user_id", "model", "dataset",
        "reachability_mean", "reachability_success_rate",
        "manipulation_resistance_mean", "self_influence",
        "external_influence", "aai", "aai_label",
        "diversity", "coverage", "volatility", "deceptive",
    ]

    # Resume support: load already-completed users
    completed_keys, existing_rows = _load_completed(CSV_PATH)
    if completed_keys:
        print(f"  RESUME: Found {len(completed_keys)} already-completed user rows")

    all_combo_results = []  # for summary table
    t_global = time.time()
    n_users = N_USERS
    first_combo_done = False

    # Open CSV in append mode if resuming, else write fresh
    if completed_keys:
        csvfile = open(CSV_PATH, "a", newline="")
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    else:
        csvfile = open(CSV_PATH, "w", newline="")
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    try:
        datasets = [
            ("MovieLens-1M", load_movielens_1m),
            ("Amazon-MI", load_amazon_digital_music),
        ]

        for ds_name, loader in datasets:
            print(f"\n{'='*70}")
            print(f"LOADING DATASET: {ds_name}")
            print(f"{'='*70}")
            data = loader()
            R = data["rating_matrix"]
            n_u, n_i = R.shape
            print(f"  {n_u} users, {n_i} items, density {(R>0).sum()/(n_u*n_i):.4f}")

            # Sample users — reuse existing user_ids from CSV if available
            existing_uids = set()
            for row in existing_rows:
                if row["dataset"] == ds_name:
                    existing_uids.add(int(row["user_id"]))
            active = np.where((R > 0).sum(axis=1) >= 20)[0]
            if existing_uids:
                # Use the exact same user_ids as previous runs
                user_ids = sorted(existing_uids)
                print(f"  Reusing {len(user_ids)} users from CSV (min 20 ratings)")
            else:
                sample_n = min(n_users, len(active))
                user_ids = np.random.choice(active, sample_n, replace=False).tolist()
                print(f"  Sampled {sample_n} users (min 20 ratings)")

            models = [
                ("MF", lambda: MatrixFactorization(
                    n_u, n_i, n_factors=64, n_epochs=20,
                    retrain_steps=25, batch_size=1024,
                )),
                ("NeuMF", lambda: NeuMF(
                    n_u, n_i, gmf_dim=32, mlp_dims=(64, 32, 16),
                    n_epochs=15, batch_size=1024,
                    retrain_steps=25, finetune_steps=15,
                )),
                ("LightGCN", lambda: LightGCN(
                    n_u, n_i, embed_dim=64, n_layers=3,
                    n_epochs=20, batch_size=2048,
                    retrain_steps=10,
                )),
                ("SASRec", lambda: SASRec(
                    n_u, n_i, embed_dim=64, n_heads=2,
                    n_layers=2, max_seq_len=50,
                    n_epochs=30, batch_size=256,
                    retrain_steps=10,
                )),
            ]

            for model_name, model_factory in models:
                # Check if entire combo already done
                combo_done = all(
                    (model_name, ds_name, uid) in completed_keys
                    for uid in user_ids
                )
                if combo_done:
                    print(f"\n  SKIP: {model_name}/{ds_name} — already complete in CSV")
                    # Rebuild combo_results from existing CSV rows
                    combo_results = []
                    for row in existing_rows:
                        if row["model"] == model_name and row["dataset"] == ds_name:
                            r = {k: row[k] for k in fieldnames}
                            # Convert numeric fields
                            for f in ["reachability_mean", "reachability_success_rate",
                                      "manipulation_resistance_mean", "self_influence",
                                      "external_influence", "aai", "diversity",
                                      "coverage", "volatility"]:
                                try:
                                    val = r[f]
                                    r[f] = float(val) if val and val != "inf" else (float("inf") if val == "inf" else 0.0)
                                except (ValueError, KeyError, TypeError):
                                    r[f] = 0.0
                            r["user_id"] = int(r["user_id"])
                            r["deceptive"] = int(r.get("deceptive", 0) or 0)
                            combo_results.append(r)
                    cov = combo_results[0].get("coverage", 0.0) if combo_results else 0.0
                    if isinstance(cov, str):
                        cov = float(cov)
                    odr_info = compute_odr(combo_results)
                    # Build summary from loaded data
                    aai_vals = [r["aai"] for r in combo_results if r["aai"] != float("inf")]
                    n_prob = sum(1 for r in combo_results if r["aai"] > 1.0)
                    reach_rates = [r["reachability_success_rate"] for r in combo_results]
                    reach_costs = [r["reachability_mean"] for r in combo_results
                                   if r["reachability_mean"] != float("inf")]
                    summary = {
                        "model": model_name, "dataset": ds_name,
                        "n_users": len(combo_results),
                        "reach_pct": float(np.mean(reach_rates)) * 100,
                        "reach_cost": float(np.mean(reach_costs)) if reach_costs else float("inf"),
                        "manip": float(np.mean([r["manipulation_resistance_mean"] for r in combo_results])),
                        "self_inf": float(np.mean([r["self_influence"] for r in combo_results])),
                        "aai": float(np.mean(aai_vals)) if aai_vals else float("inf"),
                        "pct_prob": n_prob / len(combo_results) * 100 if combo_results else 0,
                        "diversity": float(np.mean([r["diversity"] for r in combo_results])),
                        "volatility": float(np.mean([r["volatility"] for r in combo_results])),
                        "coverage": cov, "odr": odr_info["odr"],
                    }
                    all_combo_results.append(summary)
                    if not first_combo_done:
                        first_combo_done = True
                    continue

                print(f"\n{'='*70}")
                print(f"  MODEL: {model_name} on {ds_name} ({len(user_ids)} users)")
                print(f"{'='*70}")

                # Train
                t_train = time.time()
                model = model_factory()
                model.fit(R)
                print(f"  Training done in {time.time()-t_train:.1f}s")

                # Per-user metrics (incremental CSV write)
                combo_results = []
                t_combo = time.time()
                n_skipped = 0

                for uid in tqdm(user_ids, desc=f"  {model_name}/{ds_name}"):
                    if (model_name, ds_name, uid) in completed_keys:
                        # Load from existing rows
                        for row in existing_rows:
                            if (row["model"] == model_name and
                                row["dataset"] == ds_name and
                                int(row["user_id"]) == uid):
                                r = {k: row[k] for k in fieldnames}
                                for f in ["reachability_mean", "reachability_success_rate",
                                          "manipulation_resistance_mean", "self_influence",
                                          "external_influence", "aai", "diversity",
                                          "coverage", "volatility"]:
                                    try:
                                        r[f] = float(r[f]) if r[f] != "inf" else float("inf")
                                    except (ValueError, KeyError):
                                        r[f] = 0.0
                                r["user_id"] = int(r["user_id"])
                                r["deceptive"] = int(r.get("deceptive", 0) or 0)
                                combo_results.append(r)
                                n_skipped += 1
                                break
                        continue

                    r = compute_user_metrics(
                        model, model_name, uid, R, user_ids, n_u,
                    )
                    r["dataset"] = ds_name
                    r["coverage"] = ""  # placeholder, filled later
                    r["deceptive"] = ""  # placeholder
                    combo_results.append(r)

                    # Incremental CSV write (without coverage/deceptive for now)
                    row = {k: r.get(k, "") for k in fieldnames}
                    writer.writerow(row)
                    csvfile.flush()

                if n_skipped:
                    print(f"  (Resumed: {n_skipped} users loaded from CSV)")

                # Coverage (population-level)
                cov = catalog_coverage(model, user_ids, R, K)
                for r in combo_results:
                    r["coverage"] = cov

                # ODR
                odr_info = compute_odr(combo_results)

                elapsed_combo = time.time() - t_combo
                print(f"\n  {model_name}/{ds_name} done in {elapsed_combo:.1f}s")
                print(f"  ODR: {odr_info['odr']:.4f} "
                      f"({odr_info['n_deceptive']}/{odr_info['n_autonomous']} deceptive)")

                # Summary stats
                aai_vals = [r["aai"] for r in combo_results if r["aai"] != float("inf")]
                n_prob = sum(1 for r in combo_results if r["aai"] > 1.0)
                reach_rates = [r["reachability_success_rate"] for r in combo_results]
                reach_costs = [r["reachability_mean"] for r in combo_results
                               if r["reachability_mean"] != float("inf")]

                summary = {
                    "model": model_name,
                    "dataset": ds_name,
                    "n_users": len(user_ids),
                    "reach_pct": float(np.mean(reach_rates)) * 100,
                    "reach_cost": float(np.mean(reach_costs)) if reach_costs else float("inf"),
                    "manip": float(np.mean([r["manipulation_resistance_mean"] for r in combo_results])),
                    "self_inf": float(np.mean([r["self_influence"] for r in combo_results])),
                    "aai": float(np.mean(aai_vals)) if aai_vals else float("inf"),
                    "pct_prob": n_prob / len(user_ids) * 100,
                    "diversity": float(np.mean([r["diversity"] for r in combo_results])),
                    "volatility": float(np.mean([r["volatility"] for r in combo_results])),
                    "coverage": cov,
                    "odr": odr_info["odr"],
                }
                all_combo_results.append(summary)

                # Adaptive user count: after first combo, check if too slow
                if not first_combo_done:
                    first_combo_done = True
                    elapsed_total = time.time() - t_global
                    projected = elapsed_total * 8  # 4 models × 2 datasets
                    if projected > MAX_TOTAL_HOURS * 3600:
                        new_n = max(100, n_users // 2)
                        if new_n < n_users:
                            print(f"\n  WARNING: Projected total {projected/3600:.1f}h > "
                                  f"{MAX_TOTAL_HOURS}h limit.")
                            print(f"  Reducing from {n_users} to {new_n} users for "
                                  f"remaining combos.")
                            n_users = new_n

    finally:
        csvfile.close()

    # Rewrite CSV with coverage and deceptive columns filled
    _rewrite_csv_final(CSV_PATH, fieldnames, all_combo_results)

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print(f"  CSV rewritten with final coverage & deceptive columns.")
    print(f"{'='*100}")
    print(f"\n\n{'='*100}")
    print("PHASE 3 SUMMARY TABLE")
    print(f"{'='*100}")

    header = (f"{'Model':<8} | {'Dataset':<14} | {'N':>4} | {'Reach%':>7} | "
              f"{'Cost':>6} | {'Manip':>6} | {'Self':>6} | {'AAI':>6} | "
              f"{'%Prob':>6} | {'Div':>6} | {'Vol':>6} | {'Cov':>6} | {'ODR':>6}")
    print(header)
    print("-" * len(header))

    for s in all_combo_results:
        cost_str = f"{s['reach_cost']:.2f}" if s["reach_cost"] != float("inf") else "N/A"
        aai_str = f"{s['aai']:.4f}" if s["aai"] != float("inf") else "N/A"
        print(f"{s['model']:<8} | {s['dataset']:<14} | {s['n_users']:>4} | "
              f"{s['reach_pct']:>6.1f}% | {cost_str:>6} | {s['manip']:>6.4f} | "
              f"{s['self_inf']:>6.4f} | {aai_str:>6} | {s['pct_prob']:>5.1f}% | "
              f"{s['diversity']:>6.4f} | {s['volatility']:>6.4f} | "
              f"{s['coverage']*100:>5.1f}% | {s['odr']*100:>5.1f}%")

    elapsed = time.time() - t_global
    print(f"\n{'='*100}")
    print(f"PHASE 3 COMPLETE — {elapsed/60:.1f} min total")
    print(f"Results saved to {CSV_PATH}")
    print(f"{'='*100}")


if __name__ == "__main__":
    run_experiment()

"""CausalLens — main audit class."""

import numpy as np
from tqdm import tqdm

from causallens.recommender import Recommender
from causallens.metrics.reachability import reachability_cost
from causallens.metrics.manipulation import manipulation_resistance
from causallens.metrics.aai import autonomy_asymmetry_index
from causallens.metrics.odr import observational_deception_rate
from causallens.metrics.observational import compute_observational_metrics


class CausalLens:
    """Causal autonomy auditor for recommender systems.

    Usage:
        lens = CausalLens(recommender, rating_matrix)
        reach = lens.reachability(user_ids, target_items)
        manip = lens.manipulation_resistance(victim_ids, adversary_ids)
        report = lens.audit(user_sample=200)
    """

    def __init__(self, recommender: Recommender, rating_matrix: np.ndarray, k: int = 10):
        self.recommender = recommender
        self.rating_matrix = rating_matrix
        self.k = k
        self.n_users, self.n_items = rating_matrix.shape

    def reachability(
        self,
        user_ids: list[int] | None = None,
        target_items: list[int] | None = None,
        mode: str = "whitebox",
        max_budget: int = 20,
        n_targets_per_user: int = 5,
        **kwargs,
    ) -> dict:
        """Compute reachability costs for users × target items.

        Args:
            user_ids: Users to audit. If None, sample 200.
            target_items: Items to target. If None, sample random unreachable items.
            mode: "whitebox" or "blackbox".
            max_budget: Max perturbation budget.
            n_targets_per_user: Number of target items per user if target_items is None.

        Returns:
            dict with per-user results and summary statistics.
        """
        if user_ids is None:
            user_ids = self._sample_users()

        results = []
        for uid in tqdm(user_ids, desc="Reachability"):
            # Pick target items not in current top-k
            current_topk = set(self.recommender.get_recommendations(uid, self.rating_matrix, self.k).tolist())
            if target_items is not None:
                targets = [t for t in target_items if t not in current_topk]
            else:
                # Sample random items not in top-k and not rated
                unrated = np.where(self.rating_matrix[uid] == 0)[0]
                available = [i for i in unrated if i not in current_topk]
                if len(available) == 0:
                    continue
                targets = np.random.choice(
                    available,
                    min(n_targets_per_user, len(available)),
                    replace=False,
                ).tolist()

            for target in targets:
                result = reachability_cost(
                    self.recommender, uid, target, self.rating_matrix,
                    k=self.k, max_budget=max_budget, mode=mode, **kwargs,
                )
                result["user_id"] = uid
                result["target_item"] = target
                results.append(result)

        costs = [r["cost"] for r in results if r["success"]]
        success_rate = sum(1 for r in results if r["success"]) / max(len(results), 1)

        return {
            "results": results,
            "mean_cost": float(np.mean(costs)) if costs else float("inf"),
            "median_cost": float(np.median(costs)) if costs else float("inf"),
            "success_rate": success_rate,
            "n_users": len(user_ids),
            "n_evaluations": len(results),
        }

    def manipulation_resistance(
        self,
        victim_ids: list[int] | None = None,
        adversary_ids: list[int] | None = None,
        epsilon: int = 10,
        mode: str = "whitebox",
        n_adversaries_per_victim: int = 3,
        **kwargs,
    ) -> dict:
        """Compute manipulation resistance for victim × adversary pairs.

        Args:
            victim_ids: Victims to audit. If None, sample 200.
            adversary_ids: Adversary users. If None, sample randomly.
            epsilon: Adversary budget.
            mode: "whitebox" or "blackbox".
            n_adversaries_per_victim: Adversaries per victim if adversary_ids is None.

        Returns:
            dict with per-pair results and summary statistics.
        """
        if victim_ids is None:
            victim_ids = self._sample_users()

        results = []
        for vid in tqdm(victim_ids, desc="Manipulation resistance"):
            if adversary_ids is not None:
                advs = [a for a in adversary_ids if a != vid]
            else:
                all_others = [u for u in range(self.n_users) if u != vid]
                advs = np.random.choice(
                    all_others,
                    min(n_adversaries_per_victim, len(all_others)),
                    replace=False,
                ).tolist()

            for aid in advs:
                result = manipulation_resistance(
                    self.recommender, vid, aid, self.rating_matrix,
                    k=self.k, epsilon=epsilon, mode=mode, **kwargs,
                )
                result["victim_id"] = vid
                result["adversary_id"] = aid
                results.append(result)

        displacements = [r["displacement"] for r in results]

        return {
            "results": results,
            "mean_displacement": float(np.mean(displacements)) if displacements else 0.0,
            "median_displacement": float(np.median(displacements)) if displacements else 0.0,
            "max_displacement": float(np.max(displacements)) if displacements else 0.0,
            "n_victims": len(victim_ids),
            "n_evaluations": len(results),
            "epsilon": epsilon,
        }

    def aai(
        self,
        user_ids: list[int] | None = None,
        epsilon: int = 10,
        mode: str = "blackbox",
        n_adversaries: int = 3,
        **kwargs,
    ) -> dict:
        """Compute Autonomy Asymmetry Index for users.

        Args:
            user_ids: Users to audit. If None, sample 200.
            epsilon: Budget for self and adversary perturbations.
            mode: "whitebox" or "blackbox".
            n_adversaries: Number of adversaries per user.

        Returns:
            dict with per-user AAI results and summary statistics.
        """
        if user_ids is None:
            user_ids = self._sample_users()

        results = []
        for uid in tqdm(user_ids, desc="AAI"):
            result = autonomy_asymmetry_index(
                self.recommender, uid, self.rating_matrix,
                k=self.k, epsilon=epsilon, n_adversaries=n_adversaries,
                mode=mode, **kwargs,
            )
            results.append(result)

        aai_values = [r["aai"] for r in results if r["aai"] != float("inf")]
        n_problematic = sum(1 for r in results if r["aai"] > 1.0)

        return {
            "results": results,
            "mean_aai": float(np.mean(aai_values)) if aai_values else float("inf"),
            "median_aai": float(np.median(aai_values)) if aai_values else float("inf"),
            "n_problematic": n_problematic,
            "n_users": len(user_ids),
            "epsilon": epsilon,
        }

    def odr(
        self,
        user_ids: list[int] | None = None,
        max_budget: int = 20,
        reachability_mode: str = "blackbox",
        **kwargs,
    ) -> dict:
        """Compute Observational Deception Rate.

        Args:
            user_ids: Users to audit. If None, sample 200.
            max_budget: Max reachability budget.
            reachability_mode: "whitebox" or "blackbox".

        Returns:
            ODR results dict.
        """
        if user_ids is None:
            user_ids = self._sample_users()

        return observational_deception_rate(
            self.recommender, user_ids, self.rating_matrix,
            k=self.k, max_budget=max_budget,
            reachability_mode=reachability_mode, **kwargs,
        )

    def observational(
        self,
        user_ids: list[int] | None = None,
    ) -> dict:
        """Compute observational baseline metrics."""
        if user_ids is None:
            user_ids = self._sample_users()
        return compute_observational_metrics(
            self.recommender, user_ids, self.rating_matrix, k=self.k,
        )

    def audit(
        self,
        user_sample: int = 200,
        epsilon: int = 10,
        mode: str = "whitebox",
        **kwargs,
    ) -> dict:
        """Run full autonomy audit: all metrics.

        Returns combined results dict.
        """
        user_ids = self._sample_users(user_sample)

        reach = self.reachability(user_ids=user_ids, mode=mode, **kwargs)
        manip = self.manipulation_resistance(victim_ids=user_ids, epsilon=epsilon, mode=mode, **kwargs)
        aai_results = self.aai(user_ids=user_ids, epsilon=epsilon, mode=mode, **kwargs)
        obs = self.observational(user_ids=user_ids)
        odr_results = self.odr(user_ids=user_ids, reachability_mode=mode, **kwargs)

        return {
            "reachability": reach,
            "manipulation_resistance": manip,
            "aai": aai_results,
            "observational": obs,
            "odr": odr_results,
            "n_users": self.n_users,
            "n_items": self.n_items,
            "k": self.k,
            "epsilon": epsilon,
            "mode": mode,
        }

    def _sample_users(self, n: int = 200) -> list[int]:
        """Sample n users with at least some ratings."""
        active = np.where((self.rating_matrix > 0).sum(axis=1) >= 20)[0]
        if len(active) <= n:
            return active.tolist()
        return np.random.choice(active, n, replace=False).tolist()

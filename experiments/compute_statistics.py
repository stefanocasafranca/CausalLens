"""Compute bootstrap CIs, Wilson intervals, Fisher's exact tests, and
ODR threshold sensitivity from phase3_results.csv.

Run:
  PYTHONPATH=. python experiments/compute_statistics.py

Outputs exact numbers for pasting into the paper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "phase3_results.csv"

COMBOS = [
    ("MF", "MovieLens-1M"),
    ("NeuMF", "MovieLens-1M"),
    ("LightGCN", "MovieLens-1M"),
    ("SASRec", "MovieLens-1M"),
    ("MF", "Amazon-MI"),
    ("NeuMF", "Amazon-MI"),
    ("LightGCN", "Amazon-MI"),
    ("SASRec", "Amazon-MI"),
]


def combo_label(m: str, d: str) -> str:
    short = "ML-1M" if d == "MovieLens-1M" else "Amazon-MI"
    return f"{m} / {short}"


def bootstrap_ci(data: np.ndarray, stat_fn=np.mean, n_boot: int = 10_000,
                 seed: int = 42, alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high)."""
    rng = np.random.RandomState(seed)
    point = float(stat_fn(data))
    boot = np.array([stat_fn(rng.choice(data, size=len(data), replace=True))
                     for _ in range(n_boot)])
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return point, lo, hi


def wilson_ci(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Wilson score interval for a proportion. Returns (p_hat, lo, hi)."""
    if total == 0:
        return 0.0, 0.0, 0.0
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return p, max(0.0, centre - spread), min(1.0, centre + spread)


def odr_for_combo(sub: pd.DataFrame) -> tuple[int, int]:
    """Return (n_trapped_and_auto, n_auto)."""
    med_div = float(np.median(sub["diversity"]))
    med_vol = float(np.median(sub["volatility"]))
    auto = (sub["diversity"] > med_div) & (sub["volatility"] > med_vol)
    auto_sub = sub[auto]
    if len(auto_sub) == 0:
        return 0, 0
    trapped = int((auto_sub["reachability_success_rate"] == 0.0).sum())
    return trapped, len(auto_sub)


def odr_at_percentile(sub: pd.DataFrame, pct: float) -> tuple[float, int, int]:
    """ODR using diversity > pct-ile AND volatility > pct-ile thresholds."""
    div_thresh = float(np.percentile(sub["diversity"], pct))
    vol_thresh = float(np.percentile(sub["volatility"], pct))
    auto = (sub["diversity"] > div_thresh) & (sub["volatility"] > vol_thresh)
    auto_sub = sub[auto]
    if len(auto_sub) == 0:
        return float("nan"), 0, 0
    trapped = int((auto_sub["reachability_success_rate"] == 0.0).sum())
    return trapped / len(auto_sub), trapped, len(auto_sub)


def main():
    df = pd.read_csv(CSV_PATH)

    print("=" * 78)
    print("BOOTSTRAP 95% CIs FOR MANIPULATION DISPLACEMENT (10,000 resamples, seed=42)")
    print("=" * 78)
    for m, d in COMBOS:
        sub = df[(df.model == m) & (df.dataset == d)]
        vals = sub["manipulation_resistance_mean"].dropna().values
        pt, lo, hi = bootstrap_ci(vals)
        print(f"  {combo_label(m, d):25s}  mean={pt:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    print()
    print("=" * 78)
    print("BOOTSTRAP 95% CIs FOR AAI (10,000 resamples, seed=42)")
    print("=" * 78)
    for m, d in COMBOS:
        sub = df[(df.model == m) & (df.dataset == d)]
        vals = sub["aai"].replace([np.inf, -np.inf], np.nan).dropna().values
        pt, lo, hi = bootstrap_ci(vals)
        print(f"  {combo_label(m, d):25s}  mean={pt:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    print()
    print("=" * 78)
    print("WILSON SCORE 95% CIs FOR ODR (proportion)")
    print("=" * 78)
    odr_data = {}
    for m, d in COMBOS:
        sub = df[(df.model == m) & (df.dataset == d)]
        n_trap, n_auto = odr_for_combo(sub)
        p, lo, hi = wilson_ci(n_trap, n_auto)
        odr_data[(m, d)] = (n_trap, n_auto, p, lo, hi)
        print(f"  {combo_label(m, d):25s}  ODR={100*p:.1f}%  [{100*lo:.1f}%, {100*hi:.1f}%]  ({n_trap}/{n_auto})")

    print()
    print("=" * 78)
    print("FISHER'S EXACT TEST: MF vs NeuMF ODR per dataset")
    print("=" * 78)
    for dataset in ["MovieLens-1M", "Amazon-MI"]:
        mf_trap, mf_auto, _, _, _ = odr_data[("MF", dataset)]
        mf_nontrap = mf_auto - mf_trap
        neu_trap, neu_auto, _, _, _ = odr_data[("NeuMF", dataset)]
        neu_nontrap = neu_auto - neu_trap
        table = [[mf_trap, mf_nontrap], [neu_trap, neu_nontrap]]
        oddsratio, p = stats.fisher_exact(table)
        short = "ML-1M" if dataset == "MovieLens-1M" else "Amazon-MI"
        print(f"  {short:12s}  table={table}  OR={oddsratio:.3f}  p={p:.4f}")

    print()
    print("=" * 78)
    print("MANN-WHITNEY U + EFFECT SIZE: MF vs NeuMF manipulation displacement")
    print("=" * 78)
    for dataset in ["MovieLens-1M", "Amazon-MI"]:
        mf_sub = df[(df.model == "MF") & (df.dataset == dataset)]
        neu_sub = df[(df.model == "NeuMF") & (df.dataset == dataset)]
        mf_vals = mf_sub["manipulation_resistance_mean"].dropna().values
        neu_vals = neu_sub["manipulation_resistance_mean"].dropna().values
        U, p = stats.mannwhitneyu(mf_vals, neu_vals, alternative="two-sided")
        n1, n2 = len(mf_vals), len(neu_vals)
        r = 1 - 2 * U / (n1 * n2)  # rank-biserial correlation
        short = "ML-1M" if dataset == "MovieLens-1M" else "Amazon-MI"
        print(f"  {short:12s}  U={U:.0f}  p={p:.2e}  n1={n1} n2={n2}  rank-biserial r={r:.3f}")

    # Combined across datasets
    mf_all = df[df.model == "MF"]["manipulation_resistance_mean"].dropna().values
    neu_all = df[df.model == "NeuMF"]["manipulation_resistance_mean"].dropna().values
    U, p = stats.mannwhitneyu(mf_all, neu_all, alternative="two-sided")
    n1, n2 = len(mf_all), len(neu_all)
    r = 1 - 2 * U / (n1 * n2)
    print(f"  {'Combined':12s}  U={U:.0f}  p={p:.2e}  n1={n1} n2={n2}  rank-biserial r={r:.3f}")

    print()
    print("=" * 78)
    print("BASE UNREACHABILITY vs ODR + CHI-SQUARED INDEPENDENCE TEST")
    print("=" * 78)
    from scipy.stats import chi2_contingency
    for m, d in COMBOS:
        sub = df[(df.model == m) & (df.dataset == d)]
        base_trapped = int((sub["reachability_success_rate"] == 0).sum())
        base_rate = base_trapped / len(sub)
        n_trap, n_auto = odr_for_combo(sub)
        odr_val = n_trap / n_auto if n_auto > 0 else 0
        med_div = float(np.median(sub["diversity"]))
        med_vol = float(np.median(sub["volatility"]))
        auto = (sub["diversity"] > med_div) & (sub["volatility"] > med_vol)
        trapped = sub["reachability_success_rate"] == 0
        a = int(( auto & trapped).sum())
        b = int(( auto & ~trapped).sum())
        c = int((~auto & trapped).sum())
        dd = int((~auto & ~trapped).sum())
        chi2, p, _, _ = chi2_contingency([[a,b],[c,dd]])
        print(f"  {combo_label(m, d):25s}  base={100*base_rate:.1f}%  ODR={100*odr_val:.1f}%  "
              f"ratio={odr_val/base_rate:.2f}x  chi2={chi2:.2f}  p={p:.4f}")

    print()
    print("=" * 78)
    print("ODR THRESHOLD SENSITIVITY (percentile sweep)")
    print("=" * 78)
    for pct in [25, 50, 75, 90]:
        print(f"\n  Threshold: {pct}th percentile")
        for m, d in COMBOS:
            sub = df[(df.model == m) & (df.dataset == d)]
            odr_val, n_trap, n_auto = odr_at_percentile(sub, pct)
            print(f"    {combo_label(m, d):25s}  ODR={100*odr_val:.1f}%  ({n_trap}/{n_auto})")

    print()
    print("=" * 78)
    print("LATEX SNIPPETS FOR TABLE 1")
    print("=" * 78)
    for m, d in COMBOS:
        sub = df[(df.model == m) & (df.dataset == d)]
        # Reachability (finite only)
        r_vals = pd.to_numeric(sub["reachability_mean"], errors="coerce")
        r_finite = r_vals.replace([np.inf, -np.inf], np.nan).dropna().values
        r_pt, r_lo, r_hi = bootstrap_ci(r_finite) if len(r_finite) > 1 else (float("nan"),)*3

        # Manipulation
        m_vals = sub["manipulation_resistance_mean"].dropna().values
        m_pt, m_lo, m_hi = bootstrap_ci(m_vals)

        # AAI
        a_vals = sub["aai"].replace([np.inf, -np.inf], np.nan).dropna().values
        a_pt, a_lo, a_hi = bootstrap_ci(a_vals)

        # ODR
        n_trap, n_auto = odr_for_combo(sub)
        odr_p, odr_lo, odr_hi = wilson_ci(n_trap, n_auto)

        lbl = combo_label(m, d)
        print(f"  {lbl} & 200 & "
              f"${r_pt:.2f}\\,[{r_lo:.2f},\\,{r_hi:.2f}]$ & "
              f"${m_pt:.3f}\\,[{m_lo:.3f},\\,{m_hi:.3f}]$ & "
              f"${a_pt:.3f}\\,[{a_lo:.3f},\\,{a_hi:.3f}]$ & "
              f"${100*odr_p:.1f}\\,[{100*odr_lo:.1f},\\,{100*odr_hi:.1f}]$ \\\\")


if __name__ == "__main__":
    main()

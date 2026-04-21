# BUILD_SCRIPT.md

## Prompts Up to date with Output

CausalLens is an open-source Python toolkit for causal autonomy auditing of recommender systems, targeting ACM RecSys 2026. The project implements four core metrics: Reachability Cost (minimum perturbation to push a target item into top-k via two-phase whitebox gradient descent with binary search or blackbox greedy coordinate search), Manipulation Resistance (maximum Jaccard displacement an adversary can cause via gradient ascent with rank-1 V perturbation or blackbox hill-climbing), Autonomy Asymmetry Index (ratio of max adversary displacement to self-influence displacement, with healthy/borderline/problematic labels), and Observational Deception Rate (fraction of users who appear autonomous by observational metrics but are causally trapped when tested with reachability). Three observational baselines are implemented: intra-list diversity (average pairwise Jaccard distance of user profiles within top-k), catalog coverage (fraction of items appearing in any user's top-k), and recommendation volatility (Jaccard displacement after small random rating perturbation). Phase 1 delivers a from-scratch Matrix Factorization model with SGD training and a differentiable scoring path using soft sigmoid-weighted ridge regression for white-box gradients. The MF differentiable path uses sigmoid((r-0.5)*20) weighting so unrated items get near-zero weight while perturbations smoothly introduce new ratings. Validation on MovieLens-1M with 10 users confirms: 13% reachability success (random targets at ranks 500-3000 are genuinely unreachable), self-influence 0.75-1.0, MF correctly shows zero manipulation vulnerability (frozen item factors make each user's scores depend only on their own ratings), diversity 0.66-0.85, volatility 0.0-0.33, coverage 2.3%. Phase 2 adds NeuMF (from scratch, GMF+MLP dual-path neural collaborative filtering) and SASRec (RecBole wrapper, skeleton) with cross-user coupling: adversary perturbation triggers warm-start retrain of shared item embeddings, propagating effects to all users' scores. NeuMF uses step-limited retraining (50 SGD steps, not full epochs) for practical blackbox search, and cached scratch model for fast user fine-tuning. MF is also updated with retrain support (retrain_steps=100) so cross-user manipulation effects can be measured when item factors are allowed to shift. Amazon Musical Instruments 5-core data loader is added (HuggingFace Amazon Reviews 2023 format). NeuMF validation on MovieLens-1M with 10 users confirms: mean manipulation displacement 0.60 (max 0.95), demonstrating real cross-user coupling unlike frozen-V MF; self-influence 0.57-1.0; mean AAI 0.78 with 1/10 users problematic (AAI=1.15, adversary influence exceeds self-influence); diversity 0.71-0.86; volatility 0.24-0.71 (higher than MF); coverage 1.6%. Phase 3 runs the full experiment pipeline: 2 models (MF, NeuMF) x 2 datasets (MovieLens-1M, Amazon-MI) x 200 users each, computing all metrics per-user with CSV-based resume, incremental writes, and adaptive timeouts. Results: MF/ML-1M ODR=68.3% (6.5% reachability, 0.45 manipulation, 0.72 self-influence), NeuMF/ML-1M ODR=90.4% (0.9% reachability, 0.52 manipulation, 0.78 self-influence), MF/Amazon-MI ODR=88.0% (3.6% reachability, 0.58 manipulation, 0.91 self-influence), NeuMF/Amazon-MI ODR=93.9% (0.6% reachability, 0.77 manipulation, 0.97 self-influence). NeuMF consistently shows higher ODR than MF on both datasets, confirming that cross-user coupling via shared neural embeddings creates more observational deception — users appear autonomous but are causally trapped. Total runtime: 329.5 minutes for 800 user evaluations. A public GitHub repository and Google Docs sync are connected. Phase 5 outputs are verified: all 4 PDFs valid, LaTeX table compiles, all numbers in RESULTS_SUMMARY.md match CSV. A Reachability Framing section is added: 12.4% of users (99/800) are reachable within budget-20; among reachable users, NeuMF requires fewer flips (mean 8.87) than MF (mean 18.93). The paper is written in paper/causallens_paper.tex (acmart sigconf, 6 pages) with paper/references.bib (14 entries). Sections: Introduction (Instagram cross-user influence motivation), Related Work (Sharma et al. 2024, SteerEval, Beyond the Checkbox, Dean et al., Pearl), Formal Definitions (4 definitions with full math), System Design (abstract Recommender interface, white-box/black-box paths), Experiments (MF+NeuMF, ML-1M+Amazon-MI, 800 evaluations), Results (ODR 68-94%, manipulation gap 25.4%, reachability framing), Discussion, Conclusion. Repo polish: professional README.md with badges/quick-start/citation, setup.py (pip-installable), run_all.py reproduction script with --skip-training flag. All pushed to GitHub.

## Project
**Name:** CausalLens
**Overview:** Open-source Python toolkit for causal autonomy auditing of recommender systems.

**Tech Stack:** Python, PyTorch, NumPy, Pandas, scikit-learn, RecBole (Phase 2), matplotlib, tqdm

**Structure:**
```
causallens/              — core package
  __init__.py
  core.py                — CausalLens main audit class (.reachability, .manipulation_resistance, .aai, .odr, .observational, .audit)
  recommender.py         — abstract Recommender interface
  metrics/
    __init__.py
    reachability.py      — Reachability Cost (Def 1) — whitebox + blackbox
    manipulation.py      — Manipulation Resistance (Def 2) — whitebox + blackbox
    aai.py               — Autonomy Asymmetry Index (Def 3)
    odr.py               — Observational Deception Rate (Def 4)
    observational.py     — intra-list diversity, catalog coverage, volatility
  models/
    __init__.py           — exports MatrixFactorization, NeuMF, NeuMFModule
    mf.py                — Matrix Factorization from scratch (soft sigmoid ridge regression + retrain support)
    neumf.py             — NeuMF from scratch (GMF+MLP with cross-user retrain)
    sasrec.py            — SASRec RecBole wrapper (skeleton)
  data/
    __init__.py
    movielens.py         — MovieLens-1M loader
    amazon.py            — Amazon Digital Music 5-core loader
  wrappers/
    __init__.py
  report/
    __init__.py
experiments/             — experiment scripts
  validate_neumf.py      — Phase 2 NeuMF validation (10 users, all metrics)
  run_phase3.py          — Phase 3 full experiment (200 users × 2 models × 2 datasets)
results/                 — generated outputs
  phase3_results.csv     — Phase 3 per-user results (800 rows)
  figures/               — fig1-fig4 PDFs
  tables/                — tab1-tab2 LaTeX
  RESULTS_SUMMARY.md     — auto-generated results summary
paper/                   — LaTeX source
  causallens_paper.tex   — acmart sigconf paper
  references.bib         — bibliography
validate_pipeline.py     — Phase 1 validation script
run_all.py               — full reproduction pipeline (--skip-training)
setup.py                 — pip-installable package config
README.md                — project README
CAUSALLENS_SPEC.md       — master specification
requirements.txt         — dependencies
```

**Features:**
- Matrix Factorization with SGD training, differentiable soft-sigmoid ridge-regression scoring, and warm-start retrain for cross-user manipulation
- NeuMF (Neural Matrix Factorization) with GMF+MLP dual paths, cross-user retrain via shared item embeddings, and cached scratch model for fast user fine-tuning
- SASRec via RecBole wrapper (skeleton for Phase 3)
- MovieLens-1M and Amazon Musical Instruments 5-core auto-download loaders
- Two-phase whitebox reachability: maximize target score then binary search minimum budget
- Black-box reachability via greedy coordinate search (in-place modify/restore, no matrix copies)
- Whitebox manipulation resistance with rank-1 V perturbation approximation
- Black-box manipulation resistance via hill-climbing and random search (for retrain-based models)
- Autonomy Asymmetry Index: self-influence vs adversary influence ratio
- Observational Deception Rate: surface-level vs causal autonomy comparison
- Three observational baselines: diversity, coverage, volatility
- CausalLens orchestrator with audit(), reachability(), manipulation_resistance(), aai(), odr(), observational()
- Phase 3 full experiment pipeline with CSV-based resume, incremental per-user writes, and adaptive timeouts

**Commands:**
- `pip install -r requirements.txt` — install dependencies
- `PYTHONPATH=. python validate_pipeline.py` — run Phase 1 MF validation (10 users, all metrics)
- `PYTHONPATH=. python experiments/validate_neumf.py` — run Phase 2 NeuMF validation (10 users, all metrics)
- `PYTHONPATH=. python experiments/run_phase3.py` — run Phase 3 full experiment (200 users × 2 models × 2 datasets, ~5.5 hours)

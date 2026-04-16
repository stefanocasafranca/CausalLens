# BUILD_SCRIPT_FULL.md

## Prompts Up to date with Output

CausalLens is an open-source Python toolkit for causal autonomy auditing of recommender systems, targeting ACM RecSys 2026. The project implements four core metrics: Reachability Cost (minimum perturbation to push a target item into top-k via two-phase whitebox gradient descent with binary search or blackbox greedy coordinate search), Manipulation Resistance (maximum Jaccard displacement an adversary can cause via gradient ascent with rank-1 V perturbation or blackbox hill-climbing), Autonomy Asymmetry Index (ratio of max adversary displacement to self-influence displacement, with healthy/borderline/problematic labels), and Observational Deception Rate (fraction of users who appear autonomous by observational metrics but are causally trapped when tested with reachability). Three observational baselines are implemented: intra-list diversity (average pairwise Jaccard distance of user profiles within top-k), catalog coverage (fraction of items appearing in any user's top-k), and recommendation volatility (Jaccard displacement after small random rating perturbation). Phase 1 delivers a from-scratch Matrix Factorization model with SGD training and a differentiable scoring path using soft sigmoid-weighted ridge regression for white-box gradients. The MF differentiable path uses sigmoid((r-0.5)*20) weighting so unrated items get near-zero weight while perturbations smoothly introduce new ratings. Validation on MovieLens-1M with 10 users confirms: 13% reachability success (random targets at ranks 500-3000 are genuinely unreachable), self-influence 0.75-1.0, MF correctly shows zero manipulation vulnerability (frozen item factors make each user's scores depend only on their own ratings), diversity 0.66-0.85, volatility 0.0-0.33, coverage 2.3%. Phase 2 adds NeuMF (from scratch, GMF+MLP dual-path neural collaborative filtering) and SASRec (RecBole wrapper, skeleton) with cross-user coupling: adversary perturbation triggers warm-start retrain of shared item embeddings, propagating effects to all users' scores. NeuMF uses step-limited retraining (50 SGD steps, not full epochs) for practical blackbox search, and cached scratch model for fast user fine-tuning. MF is also updated with retrain support (retrain_steps=100) so cross-user manipulation effects can be measured when item factors are allowed to shift. Amazon Musical Instruments 5-core data loader is added (HuggingFace Amazon Reviews 2023 format). NeuMF validation on MovieLens-1M with 10 users confirms: mean manipulation displacement 0.60 (max 0.95), demonstrating real cross-user coupling unlike frozen-V MF; self-influence 0.57-1.0; mean AAI 0.78 with 1/10 users problematic (AAI=1.15, adversary influence exceeds self-influence); diversity 0.71-0.86; volatility 0.24-0.71 (higher than MF); coverage 1.6%. Phase 3 runs the full experiment pipeline: 2 models (MF, NeuMF) x 2 datasets (MovieLens-1M, Amazon-MI) x 200 users each, computing all metrics per-user with CSV-based resume, incremental writes, and adaptive timeouts. Results: MF/ML-1M ODR=68.3% (6.5% reachability, 0.45 manipulation, 0.72 self-influence), NeuMF/ML-1M ODR=90.4% (0.9% reachability, 0.52 manipulation, 0.78 self-influence), MF/Amazon-MI ODR=88.0% (3.6% reachability, 0.58 manipulation, 0.91 self-influence), NeuMF/Amazon-MI ODR=93.9% (0.6% reachability, 0.77 manipulation, 0.97 self-influence). NeuMF consistently shows higher ODR than MF on both datasets, confirming that cross-user coupling via shared neural embeddings creates more observational deception — users appear autonomous but are causally trapped. Total runtime: 329.5 minutes for 800 user evaluations. A public GitHub repository and Google Docs sync are connected.

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
validate_pipeline.py     — Phase 1 validation script
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

## Prompts RAW

1. /build_script
<!-- By: stefanocasafranca | 2026-04-13 -->

2. Create a new public GitHub repo called CausalLens. Then create CAUSALLENS_SPEC.md in the project root with the full project specification. This is the master specification for the entire project. Start Phase 1: build the MF model, MovieLens-1M loader, and reachability metric following the structure above.
<!-- By: stefanocasafranca | 2026-04-13 -->
<!-- Rephrased prompt for "Prompts Up to date with Output": ADD: "CausalLens is an open-source Python toolkit for causal autonomy auditing of recommender systems, targeting ACM RecSys 2026. The project implements four core metrics: Reachability Cost (minimum perturbation to push a target item into top-k), Manipulation Resistance (maximum Jaccard displacement an adversary can cause), Autonomy Asymmetry Index (ratio of external to self-influence), and Observational Deception Rate (fraction of observationally autonomous users who are causally trapped). Phase 1 delivers a from-scratch Matrix Factorization model with SGD training and a differentiable scoring path for white-box gradients, a MovieLens-1M data loader with filtering and contiguous ID remapping, white-box reachability via projected gradient descent and black-box via greedy coordinate search, white-box manipulation resistance via gradient ascent on a Jaccard surrogate and black-box via hill-climbing, and a CausalLens audit class that orchestrates all metrics with user sampling. The spec file CAUSALLENS_SPEC.md is created as the master reference. A public GitHub repository is connected for automatic commits." -->

3. Read CAUSALLENS_SPEC.md. Continue Phase 1. Implement the Autonomy Asymmetry Index (Definition 3) in causallens/metrics/aai.py and the Observational Deception Rate (Definition 4) in causallens/metrics/odr.py. Also implement the observational baselines in causallens/metrics/observational.py: intra-list diversity, catalog coverage, and recommendation volatility. Then run a quick validation: train MF on MovieLens-1M, sample 10 users, compute reachability, manipulation resistance, AAI, and all three observational baselines. Print the results as a table.
<!-- By: stefanocasafranca | 2026-04-13 -->
<!-- Rephrased prompt for "Prompts Up to date with Output":
  CHANGED: "Reachability Cost (minimum perturbation to push a target item into top-k)" → "Reachability Cost (minimum perturbation to push a target item into top-k via two-phase whitebox gradient descent with binary search or blackbox greedy coordinate search)"
  CHANGED: "Manipulation Resistance (maximum Jaccard displacement an adversary can cause)" → "Manipulation Resistance (maximum Jaccard displacement an adversary can cause via gradient ascent with rank-1 V perturbation or blackbox hill-climbing)"
  CHANGED: "Autonomy Asymmetry Index (ratio of external to self-influence)" → "Autonomy Asymmetry Index (ratio of max adversary displacement to self-influence displacement, with healthy/borderline/problematic labels)"
  CHANGED: "Observational Deception Rate (fraction of observationally autonomous users who are causally trapped)" → "Observational Deception Rate (fraction of users who appear autonomous by observational metrics but are causally trapped when tested with reachability)"
  ADD: "Three observational baselines are implemented: intra-list diversity (average pairwise Jaccard distance of user profiles within top-k), catalog coverage (fraction of items appearing in any user's top-k), and recommendation volatility (Jaccard displacement after small random rating perturbation)."
  CHANGED: "a differentiable scoring path for white-box gradients" → "a differentiable scoring path using soft sigmoid-weighted ridge regression for white-box gradients"
  ADD: "The MF differentiable path uses sigmoid((r-0.5)*20) weighting so unrated items get near-zero weight while perturbations smoothly introduce new ratings."
  ADD: "Validation on MovieLens-1M with 10 users confirms: 13% reachability success (random targets at ranks 500-3000 are genuinely unreachable), self-influence 0.75-1.0, MF correctly shows zero manipulation vulnerability (frozen item factors make each user's scores depend only on their own ratings), diversity 0.66-0.85, volatility 0.0-0.33, coverage 2.3%."
  ADD: "A public GitHub repository and Google Docs sync are connected."
-->

4. Read CAUSALLENS_SPEC.md. Start Phase 2. Important context from Phase 1: MF with frozen item factors shows 0 adversary influence because scores depend only on the user's own ratings. We need models with cross-user coupling to make AAI meaningful. Do these steps: 1. Add NeuMF and SASRec via RecBole. Wrap each in our Recommender interface. For manipulation resistance in these models, the adversary perturbation must trigger a model retrain (or partial update) so that cross-user effects propagate — changing user A's ratings should affect the learned embeddings which then affect user B's scores. 2. Add the Amazon Digital Music data loader in causallens/data/amazon.py (download from https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/ — use the 5-core Digital Music ratings). 3. Also update the MF manipulation resistance to retrain the full model (both user AND item factors) after adversary perturbation, so we can measure whether MF shows cross-user influence when item embeddings are allowed to shift. 4. Run the same 10-user validation on NeuMF with MovieLens-1M: reachability, manipulation resistance, AAI, and observational baselines. Print results so we can compare against MF.
<!-- By: stefanocasafranca | 2026-04-14 -->
<!-- Rephrased prompt for "Prompts Up to date with Output":
  ADD: "Phase 2 adds NeuMF (from scratch, GMF+MLP dual-path neural collaborative filtering) and SASRec (RecBole wrapper, skeleton) with cross-user coupling: adversary perturbation triggers warm-start retrain of shared item embeddings, propagating effects to all users' scores."
  ADD: "NeuMF uses step-limited retraining (50 SGD steps, not full epochs) for practical blackbox search, and cached scratch model for fast user fine-tuning."
  ADD: "MF is also updated with retrain support (retrain_steps=100) so cross-user manipulation effects can be measured when item factors are allowed to shift."
  ADD: "Amazon Digital Music 5-core data loader is added for Phase 3 experiments."
  ADD: "NeuMF validation on MovieLens-1M with 10 users confirms: mean manipulation displacement 0.60 (max 0.95), demonstrating real cross-user coupling unlike frozen-V MF; self-influence 0.57-1.0; mean AAI 0.78 with 1/10 users problematic (AAI=1.15, adversary influence exceeds self-influence); diversity 0.71-0.86; volatility 0.24-0.71 (higher than MF); coverage 1.6%."
-->

5. Run Phase 3: full experiment pipeline with 2 models (MF, NeuMF) × 2 datasets (MovieLens-1M, Amazon Musical Instruments) × 200 users. Compute all metrics per-user, derive ODR, output CSV and summary table. Use CSV-based resume so crashed runs can continue. Switch Amazon dataset from Digital Music (404) to Musical Instruments 5-core from HuggingFace Amazon Reviews 2023.
<!-- By: stefanocasafranca | 2026-04-15 -->
<!-- Rephrased prompt for "Prompts Up to date with Output":
  CHANGED: "Amazon Digital Music 5-core data loader is added for Phase 3 experiments." → "Amazon Musical Instruments 5-core data loader is added (HuggingFace Amazon Reviews 2023 format)."
  ADD: "Phase 3 runs the full experiment pipeline: 2 models (MF, NeuMF) x 2 datasets (MovieLens-1M, Amazon-MI) x 200 users each, computing all metrics per-user with CSV-based resume, incremental writes, and adaptive timeouts."
  ADD: "Results: MF/ML-1M ODR=68.3% (6.5% reachability, 0.45 manipulation, 0.72 self-influence), NeuMF/ML-1M ODR=90.4% (0.9% reachability, 0.52 manipulation, 0.78 self-influence), MF/Amazon-MI ODR=88.0% (3.6% reachability, 0.58 manipulation, 0.91 self-influence), NeuMF/Amazon-MI ODR=93.9% (0.6% reachability, 0.77 manipulation, 0.97 self-influence)."
  ADD: "NeuMF consistently shows higher ODR than MF on both datasets, confirming that cross-user coupling via shared neural embeddings creates more observational deception — users appear autonomous but are causally trapped."
  ADD: "Total runtime: 329.5 minutes for 800 user evaluations."
-->

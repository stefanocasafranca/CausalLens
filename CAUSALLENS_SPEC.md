PROJECT: CausalLens — an open-source Python toolkit for causal autonomy auditing of recommender systems. Target: ACM RecSys 2026 short/long paper, deadline April 21.

MATH DEFINITIONS:

Setup: U = {1,...,n} users, V = {1,...,m} items, r_i ∈ ℝ^m is user i's rating vector, R ∈ ℝ^{n×m} full rating matrix, f is the recommender function, top-k(f,i) is the set of k recommended items, d(·,·) is Jaccard distance between two top-k sets.

Definition 1 — Reachability Cost: R(i,j,k) = min_δ ‖δ‖₁ subject to rank(f,j,i|r_i+δ)≤k, δ∈C. Perturbation to user's OWN ratings. White-box: projected gradient descent. Black-box: finite differences. Interpretation: how many ratings must you change to get a target item into top-k.

Definition 2 — Manipulation Resistance: M(i,a,ε) = max_{δ_a} d(top-k(f,i|R), top-k(f,i|R+Δ_a)) subject to ‖δ_a‖₁≤ε, Δ_a modifies ONLY adversary's ratings. White-box: gradient ascent. Black-box: hill-climbing. Interpretation: how much can a stranger move YOUR feed by changing THEIR ratings.

Definition 3 — Autonomy Asymmetry Index (AAI) [NOVEL]: S(i,ε) = self-influence displacement, E(i,ε) = max adversary displacement, AAI(i,ε) = E/S. AAI<1 healthy, AAI=1 borderline, AAI>1 problematic (strangers have more power than you).

Definition 4 — Observational Deception Rate (ODR) [NOVEL]: Of users who LOOK autonomous by observational metrics (diversity, coverage, volatility), what percentage are actually trapped when tested causally. ODR = count(deceptive) / count(observationally flagged as autonomous). Thresholds at population medians.

PROJECT STRUCTURE:
causallens/ — core package
  core.py — CausalLens main class
  recommender.py — abstract Recommender interface with get_recommendations(), submit_feedback(), get_scores()
  metrics/reachability.py, manipulation.py, aai.py, odr.py, observational.py
  models/mf.py (from scratch), neumf.py, sasrec.py, lightgcn.py (via RecBole)
  wrappers/base.py (BlackBoxRecommender abstract), spotify.py
  data/movielens.py, amazon.py, preprocessing.py
  report/generator.py — Autonomy Report Card
experiments/ — run_all.py, exp_whitebox.py, exp_blackbox.py, exp_ablation.py
results/ — generated CSVs and figures

API: CausalLens(recommender, rating_matrix) with methods .reachability(), .manipulation_resistance(), .aai(), .odr(), .audit()

DESIGN DECISIONS: MF from scratch first (no RecBole for this one). Jaccard distance for d(). 200 user sample per dataset. Budget ε as discrete number of rating changes. Thresholds at median. Everything must run on a laptop.

DATASETS: MovieLens-1M, Amazon Digital Music. Spotify as stretch goal.
MODELS: MF (from scratch), NeuMF, SASRec, LightGCN (via RecBole).
DEPENDENCIES: numpy, scipy, torch, pandas, scikit-learn, recbole, matplotlib, tqdm

PHASES:
Phase 1 (Day 1-2): MF from scratch + MovieLens loader + reachability + manipulation resistance
Phase 2 (Day 3-4): RecBole models + Amazon data + black-box metrics + Spotify wrapper
Phase 3 (Day 5): Full experiments, ablation, ODR computation
Phase 4 (Day 6-7): Results, figures, paper writing

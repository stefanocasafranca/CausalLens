# BUILD_SCRIPT_FULL.md

## Prompts Up to date with Output

CausalLens is an open-source Python toolkit for causal autonomy auditing of recommender systems, targeting ACM RecSys 2026. The project implements four core metrics: Reachability Cost (minimum perturbation to push a target item into top-k), Manipulation Resistance (maximum Jaccard displacement an adversary can cause), Autonomy Asymmetry Index (ratio of external to self-influence), and Observational Deception Rate (fraction of observationally autonomous users who are causally trapped). Phase 1 delivers a from-scratch Matrix Factorization model with SGD training and a differentiable scoring path for white-box gradients, a MovieLens-1M data loader with filtering and contiguous ID remapping, white-box reachability via projected gradient descent and black-box via greedy coordinate search, white-box manipulation resistance via gradient ascent on a Jaccard surrogate and black-box via hill-climbing, and a CausalLens audit class that orchestrates all metrics with user sampling. The spec file CAUSALLENS_SPEC.md is created as the master reference. A public GitHub repository is connected for automatic commits.

## Project
**Name:** CausalLens
**Overview:** Open-source Python toolkit for causal autonomy auditing of recommender systems.

**Tech Stack:** Python, PyTorch, NumPy, Pandas, scikit-learn, RecBole (Phase 2), matplotlib, tqdm

**Structure:**
```
causallens/              — core package
  __init__.py
  core.py                — CausalLens main audit class
  recommender.py         — abstract Recommender interface
  metrics/
    __init__.py
    reachability.py      — Reachability Cost (Def 1)
    manipulation.py      — Manipulation Resistance (Def 2)
  models/
    __init__.py
    mf.py                — Matrix Factorization from scratch
  data/
    __init__.py
    movielens.py         — MovieLens-1M loader
  wrappers/
    __init__.py
  report/
    __init__.py
experiments/             — experiment scripts
results/                 — generated outputs
CAUSALLENS_SPEC.md       — master specification
requirements.txt         — dependencies
```

**Features:**
- Matrix Factorization with SGD training and differentiable ridge-regression scoring
- MovieLens-1M auto-download, filter, and rating matrix construction
- White-box reachability via projected gradient descent
- Black-box reachability via greedy coordinate search
- White-box manipulation resistance via gradient ascent on Jaccard surrogate
- Black-box manipulation resistance via hill-climbing
- CausalLens orchestrator with audit(), reachability(), manipulation_resistance()

**Commands:**
- `pip install -r requirements.txt` — install dependencies

## Prompts RAW

1. /build_script
<!-- By: stefanocasafranca | 2026-04-13 -->

2. Create a new public GitHub repo called CausalLens. Then create CAUSALLENS_SPEC.md in the project root with the full project specification. This is the master specification for the entire project. Start Phase 1: build the MF model, MovieLens-1M loader, and reachability metric following the structure above.
<!-- By: stefanocasafranca | 2026-04-13 -->
<!-- Rephrased prompt for "Prompts Up to date with Output": ADD: "CausalLens is an open-source Python toolkit for causal autonomy auditing of recommender systems, targeting ACM RecSys 2026. The project implements four core metrics: Reachability Cost (minimum perturbation to push a target item into top-k), Manipulation Resistance (maximum Jaccard displacement an adversary can cause), Autonomy Asymmetry Index (ratio of external to self-influence), and Observational Deception Rate (fraction of observationally autonomous users who are causally trapped). Phase 1 delivers a from-scratch Matrix Factorization model with SGD training and a differentiable scoring path for white-box gradients, a MovieLens-1M data loader with filtering and contiguous ID remapping, white-box reachability via projected gradient descent and black-box via greedy coordinate search, white-box manipulation resistance via gradient ascent on a Jaccard surrogate and black-box via hill-climbing, and a CausalLens audit class that orchestrates all metrics with user sampling. The spec file CAUSALLENS_SPEC.md is created as the master reference. A public GitHub repository is connected for automatic commits." -->

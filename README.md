# CausalLens

An open-source Python toolkit for causal autonomy auditing of recommender systems.

![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from causallens.models.mf import MatrixFactorization
from causallens.data.movielens import load_movielens_1m
from causallens.core import CausalLens

data = load_movielens_1m()
R = data["rating_matrix"]
model = MatrixFactorization(R.shape[0], R.shape[1], n_factors=64, n_epochs=20)
model.fit(R)
lens = CausalLens(model, R, k=10)
results = lens.audit(user_id=42)
```

## Metrics

- **Reachability Cost** — minimum rating changes needed to push a target item into a user's top-k.
- **Manipulation Resistance** — maximum Jaccard displacement an adversary can cause by changing only their own ratings.
- **Autonomy Asymmetry Index (AAI)** — ratio of adversary influence to self-influence; AAI > 1 means a stranger controls your feed more than you do.
- **Observational Deception Rate (ODR)** — fraction of users who look autonomous by diversity/volatility metrics but are causally trapped.

## Reproducing Paper Results

```bash
# Full reproduction (trains models + runs 800 user evaluations + generates figures)
PYTHONPATH=. python run_all.py

# Skip training, regenerate figures from existing CSV
PYTHONPATH=. python run_all.py --skip-training

# Individual steps
PYTHONPATH=. python experiments/run_phase3.py          # Run experiment
PYTHONPATH=. python experiments/generate_figures.py     # Generate figures + tables
```

Results are saved to `results/` (CSV, figures, tables, summary).

## Project Structure

```
causallens/           Core library
  core.py             CausalLens orchestrator
  recommender.py      Abstract Recommender interface
  metrics/            Reachability, manipulation, AAI, ODR, observational baselines
  models/             MF (from scratch), NeuMF (from scratch), SASRec (skeleton)
  data/               MovieLens-1M and Amazon Musical Instruments loaders
experiments/          Experiment scripts and figure generation
results/              Generated outputs (CSV, figures, tables)
paper/                LaTeX source for RecSys 2026 paper
```

## Citation

```bibtex
@inproceedings{casafranca2026causallens,
  author    = {Casafranca, Stefano},
  title     = {CausalLens: An Open-Source Toolkit for Causal Autonomy Auditing of Recommender Systems},
  booktitle = {Proceedings of the 20th ACM Conference on Recommender Systems},
  year      = {2026},
  publisher = {ACM},
}
```

## License

MIT

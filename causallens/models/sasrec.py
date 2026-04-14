"""SASRec wrapper via RecBole.

Self-Attentive Sequential Recommendation. Converts the rating matrix
to interaction sequences sorted by timestamp, trains via RecBole,
and wraps in our Recommender interface.

Cross-user coupling: item embeddings and attention weights are shared,
so retraining on perturbed data shifts all users' scores.
"""

import os
import tempfile
import numpy as np
import torch
import copy
from pathlib import Path

from causallens.recommender import Recommender

# Lazy imports to avoid RecBole overhead at module level
_recbole_imported = False
_run_recbole = None
_create_dataset = None
_data_preparation = None
_Interaction = None


def _ensure_recbole():
    global _recbole_imported, _run_recbole, _create_dataset, _data_preparation, _Interaction
    if _recbole_imported:
        return
    from recbole.quick_start import run_recbole
    from recbole.data import create_dataset, data_preparation
    from recbole.data.interaction import Interaction
    _run_recbole = run_recbole
    _create_dataset = create_dataset
    _data_preparation = data_preparation
    _Interaction = Interaction
    _recbole_imported = True


def _matrix_to_inter_file(rating_matrix: np.ndarray, path: str,
                          timestamps: np.ndarray | None = None):
    """Write rating matrix to RecBole .inter atomic file format."""
    users, items = np.nonzero(rating_matrix)
    ratings = rating_matrix[users, items]
    if timestamps is None:
        timestamps = np.arange(len(users))
    with open(path, "w") as f:
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for u, i, r, t in zip(users, items, ratings, timestamps):
            f.write(f"{u}\t{i}\t{r}\t{t}\n")


class SASRecRecommender(Recommender):
    """SASRec via RecBole with retrain support.

    Because RecBole training is heavyweight, manipulation resistance
    with this model uses batch-retrain: perturb adversary's ratings,
    retrain fully, measure displacement.
    """

    def __init__(self, n_users: int, n_items: int,
                 n_epochs: int = 30, embedding_size: int = 64,
                 n_layers: int = 2, n_heads: int = 2,
                 max_seq_length: int = 50, batch_size: int = 256,
                 retrain_epochs: int = 5):
        self.n_users = n_users
        self.n_items = n_items
        self.n_epochs = n_epochs
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.retrain_epochs = retrain_epochs

        self._model = None
        self._dataset = None
        self._training_matrix: np.ndarray | None = None
        self._base_state: dict | None = None
        self._tmpdir: str | None = None

    def fit(self, rating_matrix: np.ndarray, timestamps: np.ndarray | None = None,
            verbose: bool = True) -> "SASRecRecommender":
        """Train SASRec via RecBole."""
        _ensure_recbole()
        self._training_matrix = rating_matrix.copy()

        # Create temp directory for RecBole data
        self._tmpdir = tempfile.mkdtemp(prefix="causallens_sasrec_")
        dataset_name = "causallens"
        dataset_dir = os.path.join(self._tmpdir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        inter_file = os.path.join(dataset_dir, f"{dataset_name}.inter")
        _matrix_to_inter_file(rating_matrix, inter_file, timestamps)

        config_dict = {
            "model": "SASRec",
            "dataset": dataset_name,
            "data_path": self._tmpdir,
            "USER_ID_FIELD": "user_id",
            "ITEM_ID_FIELD": "item_id",
            "RATING_FIELD": "rating",
            "TIME_FIELD": "timestamp",
            "load_col": {"inter": ["user_id", "item_id", "rating", "timestamp"]},
            "MAX_ITEM_LIST_LENGTH": self.max_seq_length,
            "embedding_size": self.embedding_size,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "hidden_size": self.embedding_size,
            "inner_size": self.embedding_size * 4,
            "hidden_dropout_prob": 0.2,
            "attn_dropout_prob": 0.2,
            "hidden_act": "gelu",
            "loss_type": "CE",
            "train_batch_size": self.batch_size,
            "eval_batch_size": self.batch_size,
            "epochs": self.n_epochs,
            "learning_rate": 1e-3,
            "eval_args": {"split": {"RS": [0.8, 0.1, 0.1]}, "mode": "full", "order": "TO"},
            "metrics": ["Recall", "NDCG"],
            "topk": [10],
            "valid_metric": "Recall@10",
            "show_progress": verbose,
            "checkpoint_dir": os.path.join(self._tmpdir, "checkpoints"),
        }

        result = _run_recbole(config_dict=config_dict)
        self._model = result["model"]
        self._dataset = result["test_data"].dataset if hasattr(result, "test_data") else None
        self._base_state = copy.deepcopy(self._model.state_dict())
        return self

    def get_scores(self, user_id: int, rating_matrix: np.ndarray) -> np.ndarray:
        """Score all items for a user. Basic cached scoring."""
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # For now, return scores from cached model
        # Full retrain-based scoring is done by manipulation metric directly
        self._model.eval()
        user_tensor = torch.tensor([user_id], dtype=torch.long)
        items = torch.arange(self.n_items, dtype=torch.long)

        with torch.no_grad():
            scores = self._model.full_sort_predict(
                _Interaction({"user_id": user_tensor})
            )
        return scores.numpy()[:self.n_items]

    def retrain(self, rating_matrix: np.ndarray, n_epochs: int | None = None) -> "SASRecRecommender":
        """Warm-start retrain on modified data."""
        _ensure_recbole()
        if n_epochs is None:
            n_epochs = self.retrain_epochs

        # Write new data and retrain
        dataset_name = "causallens"
        inter_file = os.path.join(self._tmpdir, dataset_name, f"{dataset_name}.inter")
        _matrix_to_inter_file(rating_matrix, inter_file)

        # Restore base state and fine-tune
        self._model.load_state_dict(copy.deepcopy(self._base_state))
        # Training loop would go here — simplified for now
        return self

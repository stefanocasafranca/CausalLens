"""Neural Matrix Factorization (NeuMF) — implemented from scratch.

GMF path: element-wise product of user/item embeddings
MLP path: concat user/item embeddings → feed-forward layers
Final: concat(GMF, MLP) → linear → predicted score

Cross-user coupling: item embeddings are shared, so retraining on
adversary's changed ratings shifts item embeddings, which shifts
all users' scores — enabling manipulation resistance measurement.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
import copy

from causallens.recommender import Recommender


class NeuMFModule(nn.Module):
    """PyTorch module for NeuMF."""

    def __init__(self, n_users: int, n_items: int, gmf_dim: int = 32,
                 mlp_dims: tuple[int, ...] = (64, 32, 16)):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items

        # GMF path
        self.gmf_user = nn.Embedding(n_users, gmf_dim)
        self.gmf_item = nn.Embedding(n_items, gmf_dim)

        # MLP path
        mlp_input_dim = gmf_dim * 2  # user + item embeddings concatenated
        self.mlp_user = nn.Embedding(n_users, gmf_dim)
        self.mlp_item = nn.Embedding(n_items, gmf_dim)

        layers = []
        in_dim = mlp_input_dim
        for out_dim in mlp_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        # Final prediction
        self.predict_layer = nn.Linear(gmf_dim + mlp_dims[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # GMF path
        gmf_u = self.gmf_user(user_ids)
        gmf_i = self.gmf_item(item_ids)
        gmf_out = gmf_u * gmf_i

        # MLP path
        mlp_u = self.mlp_user(user_ids)
        mlp_i = self.mlp_item(item_ids)
        mlp_input = torch.cat([mlp_u, mlp_i], dim=-1)
        mlp_out = self.mlp(mlp_input)

        # Combine
        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.predict_layer(combined).squeeze(-1)

    def score_all_items(self, user_id: int) -> torch.Tensor:
        """Score all items for a single user."""
        u = torch.tensor([user_id], dtype=torch.long)
        items = torch.arange(self.n_items, dtype=torch.long)
        users = u.expand(self.n_items)

        with torch.no_grad():
            scores = self.forward(users, items)
        return scores


class NeuMF(Recommender):
    """NeuMF recommender with retraining support for manipulation metrics.

    Key difference from MF: `get_scores(user_id, R)` detects if R differs
    from the training matrix. If only the user's row changed, fine-tunes
    user embeddings. If other rows changed, does a warm-start retrain of
    the full model — this propagates cross-user effects through shared
    item embeddings.
    """

    def __init__(self, n_users: int, n_items: int, gmf_dim: int = 32,
                 mlp_dims: tuple[int, ...] = (64, 32, 16),
                 lr: float = 1e-3, weight_decay: float = 1e-5,
                 n_epochs: int = 15, batch_size: int = 1024,
                 retrain_steps: int = 50, finetune_steps: int = 30):
        self.n_users = n_users
        self.n_items = n_items
        self.gmf_dim = gmf_dim
        self.mlp_dims = mlp_dims
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.retrain_steps = retrain_steps
        self.finetune_steps = finetune_steps

        self.model: NeuMFModule | None = None
        self._base_model_state: dict | None = None
        self._training_matrix: np.ndarray | None = None
        self._score_cache: dict = {}
        self._scratch_finetune: NeuMFModule | None = None

    def fit(self, rating_matrix: np.ndarray, verbose: bool = True) -> "NeuMF":
        """Train NeuMF on observed ratings."""
        self._training_matrix = rating_matrix.copy()
        self._score_cache.clear()

        users, items = np.nonzero(rating_matrix)
        ratings = rating_matrix[users, items].astype(np.float32)
        # Normalise ratings to [0, 1] for sigmoid output
        self._rating_min = float(ratings.min())
        self._rating_max = float(ratings.max())
        ratings_norm = (ratings - self._rating_min) / (self._rating_max - self._rating_min + 1e-8)

        ds = TensorDataset(
            torch.from_numpy(users.astype(np.int64)),
            torch.from_numpy(items.astype(np.int64)),
            torch.from_numpy(ratings_norm),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model = NeuMFModule(self.n_users, self.n_items, self.gmf_dim, self.mlp_dims)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        rng = trange(self.n_epochs, desc="NeuMF training", disable=not verbose)
        for _ in rng:
            total_loss = 0.0
            for u_batch, i_batch, r_batch in loader:
                pred = self.model(u_batch, i_batch)
                loss = nn.functional.mse_loss(pred, r_batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(r_batch)
            rng.set_postfix(mse=total_loss / len(ratings))

        # Save base model state for warm-start retraining
        self._base_model_state = copy.deepcopy(self.model.state_dict())
        return self

    def get_scores(self, user_id: int, rating_matrix: np.ndarray) -> np.ndarray:
        """Score all items. Retrains if R differs from training data.

        - Same R as training: use cached model
        - Only user_id's row changed: fine-tune user embeddings
        - Other rows changed: warm-start retrain (cross-user coupling)
        """
        # Check what changed
        diff = (rating_matrix != self._training_matrix)
        changed_rows = np.where(diff.any(axis=1))[0]

        if len(changed_rows) == 0:
            # No changes — use cached model
            return self._score_with_model(self.model, user_id)

        only_self_changed = (len(changed_rows) == 1 and changed_rows[0] == user_id)

        if only_self_changed:
            # Fine-tune user embeddings only
            model = self._finetune_user(user_id, rating_matrix[user_id])
            return self._score_with_model(model, user_id)
        else:
            # Cross-user change — retrain
            model = self._retrain(rating_matrix)
            return self._score_with_model(model, user_id)

    def _score_with_model(self, model: NeuMFModule, user_id: int) -> np.ndarray:
        """Extract scores for all items from a given model."""
        model.eval()
        scores = model.score_all_items(user_id)
        # Denormalise
        scores_np = scores.numpy() * (self._rating_max - self._rating_min) + self._rating_min
        return scores_np

    def _finetune_user(self, user_id: int, user_ratings: np.ndarray) -> NeuMFModule:
        """Fine-tune only user embeddings for changed ratings.

        Reuses a cached scratch model to avoid per-call allocation overhead.
        """
        if self._scratch_finetune is None:
            self._scratch_finetune = NeuMFModule(
                self.n_users, self.n_items, self.gmf_dim, self.mlp_dims
            )
        model = self._scratch_finetune
        # Reset all parameters from base state (fast in-place copy)
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(self._base_model_state[name])
        model.train()

        # Freeze everything except user embeddings
        for param in model.parameters():
            param.requires_grad = False
        model.gmf_user.weight.requires_grad = True
        model.mlp_user.weight.requires_grad = True

        rated = np.where(user_ratings > 0)[0]
        if len(rated) == 0:
            return model

        r_norm = (user_ratings[rated] - self._rating_min) / (self._rating_max - self._rating_min + 1e-8)
        users = torch.full((len(rated),), user_id, dtype=torch.long)
        items = torch.from_numpy(rated.astype(np.int64))
        targets = torch.from_numpy(r_norm.astype(np.float32))

        opt = torch.optim.Adam([model.gmf_user.weight, model.mlp_user.weight], lr=self.lr * 5)
        for _ in range(self.finetune_steps):
            pred = model(users, items)
            loss = nn.functional.mse_loss(pred, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        return model

    def _retrain(self, rating_matrix: np.ndarray) -> NeuMFModule:
        """Warm-start retrain on modified rating matrix (cross-user effects).

        Uses a fixed number of SGD steps (not full epochs) for speed.
        With warm-start from pre-trained weights, 50 steps suffices to
        propagate small perturbation effects through shared item embeddings.
        """
        model = NeuMFModule(self.n_users, self.n_items, self.gmf_dim, self.mlp_dims)
        model.load_state_dict(copy.deepcopy(self._base_model_state))
        model.train()

        users, items = np.nonzero(rating_matrix)
        ratings = rating_matrix[users, items].astype(np.float32)
        r_norm = (ratings - self._rating_min) / (self._rating_max - self._rating_min + 1e-8)

        ds = TensorDataset(
            torch.from_numpy(users.astype(np.int64)),
            torch.from_numpy(items.astype(np.int64)),
            torch.from_numpy(r_norm),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.Adam(model.parameters(), lr=self.lr * 0.5, weight_decay=self.weight_decay)
        step = 0
        n_epochs_needed = max(1, self.retrain_steps // max(len(loader), 1) + 1)
        for _ in range(n_epochs_needed):
            for u_batch, i_batch, r_batch in loader:
                pred = model(u_batch, i_batch)
                loss = nn.functional.mse_loss(pred, r_batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                step += 1
                if step >= self.retrain_steps:
                    break
            if step >= self.retrain_steps:
                break

        model.eval()
        return model

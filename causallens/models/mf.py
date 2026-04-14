"""Matrix Factorization recommender — implemented from scratch with PyTorch.

Cross-user coupling: when get_scores detects that rows OTHER than the
queried user changed, it does a warm-start retrain of U, V, bu, bi so
that item embeddings shift — propagating adversary perturbations to all
users' scores. When only the user's own row changed, it re-solves the
user embedding via ridge regression (fast, no retrain).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from causallens.recommender import Recommender


class MatrixFactorization(Recommender):
    """ALS-style matrix factorization with SGD training.

    Factorises R ≈ U V^T + biases. Trained on observed entries only.

    After training, scores for a *perturbed* rating matrix are computed by
    solving for a new user embedding while keeping item factors fixed —
    this makes white-box gradient computation through the scoring function
    straightforward.

    For cross-user manipulation resistance, when other users' ratings change,
    the model does a warm-start retrain of all parameters (U, V, bu, bi)
    so item embeddings shift and the effects propagate to the victim.
    """

    def __init__(self, n_users: int, n_items: int, n_factors: int = 64,
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 n_epochs: int = 20, batch_size: int = 1024,
                 retrain_steps: int = 100,
                 device: str | None = None):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.retrain_steps = retrain_steps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Learnable parameters
        self.user_factors: torch.Tensor | None = None
        self.item_factors: torch.Tensor | None = None
        self.user_bias: torch.Tensor | None = None
        self.item_bias: torch.Tensor | None = None
        self.global_bias: float = 0.0
        self._training_matrix: np.ndarray | None = None
        self._base_state: dict | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, rating_matrix: np.ndarray, verbose: bool = True) -> "MatrixFactorization":
        """Train on observed (nonzero) entries of the rating matrix."""
        users, items = np.nonzero(rating_matrix)
        ratings = rating_matrix[users, items].astype(np.float32)
        self.global_bias = float(ratings.mean())

        ds = TensorDataset(
            torch.from_numpy(users.astype(np.int64)),
            torch.from_numpy(items.astype(np.int64)),
            torch.from_numpy(ratings - self.global_bias),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        # Initialise embeddings
        U = nn.Embedding(self.n_users, self.n_factors).to(self.device)
        V = nn.Embedding(self.n_items, self.n_factors).to(self.device)
        bu = nn.Embedding(self.n_users, 1).to(self.device)
        bi = nn.Embedding(self.n_items, 1).to(self.device)
        nn.init.normal_(U.weight, 0, 0.01)
        nn.init.normal_(V.weight, 0, 0.01)
        nn.init.zeros_(bu.weight)
        nn.init.zeros_(bi.weight)

        opt = torch.optim.Adam(
            list(U.parameters()) + list(V.parameters()) +
            list(bu.parameters()) + list(bi.parameters()),
            lr=self.lr, weight_decay=self.weight_decay,
        )

        rng = trange(self.n_epochs, desc="MF training", disable=not verbose)
        for _ in rng:
            total_loss = 0.0
            for u_batch, i_batch, r_batch in loader:
                u_batch = u_batch.to(self.device)
                i_batch = i_batch.to(self.device)
                r_batch = r_batch.to(self.device)

                pred = (U(u_batch) * V(i_batch)).sum(dim=1) + bu(u_batch).squeeze() + bi(i_batch).squeeze()
                loss = nn.functional.mse_loss(pred, r_batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(r_batch)
            rng.set_postfix(mse=total_loss / len(ratings))

        # Store as plain tensors (detached, on CPU for portability)
        self.user_factors = U.weight.detach().cpu()
        self.item_factors = V.weight.detach().cpu()
        self.user_bias = bu.weight.detach().cpu().squeeze()
        self.item_bias = bi.weight.detach().cpu().squeeze()

        # Save training matrix and base state for retrain support
        self._training_matrix = rating_matrix.copy()
        self._base_state = {
            "user_factors": self.user_factors.clone(),
            "item_factors": self.item_factors.clone(),
            "user_bias": self.user_bias.clone(),
            "item_bias": self.item_bias.clone(),
            "global_bias": self.global_bias,
        }
        return self

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def get_scores(self, user_id: int, rating_matrix: np.ndarray) -> np.ndarray:
        """Predict scores for all items.

        - Same R as training: use cached model with ridge regression
        - Only user_id's row changed: re-solve user embedding (fast)
        - Other rows changed: warm-start retrain all params (cross-user coupling)
        """
        if self._training_matrix is not None:
            diff = (rating_matrix != self._training_matrix)
            changed_rows = np.where(diff.any(axis=1))[0]
            if len(changed_rows) > 0:
                only_self = (len(changed_rows) == 1 and changed_rows[0] == user_id)
                if not only_self:
                    return self._retrain_and_score(user_id, rating_matrix)

        # Self-change or no change — ridge regression with current item factors
        r = torch.tensor(rating_matrix[user_id], dtype=torch.float32)
        scores = self.get_scores_differentiable(user_id, r)
        return scores.detach().numpy()

    def get_scores_differentiable(self, user_id: int, rating_vector: torch.Tensor) -> torch.Tensor:
        """Differentiable scoring path for white-box gradient computation.

        Given a (possibly perturbed) rating vector for user_id, solve for
        a new user embedding via weighted ridge regression, then score all
        items. Uses a soft sigmoid weighting so gradients flow even when
        perturbations introduce new ratings on previously-unrated items.

        Args:
            user_id: User index.
            rating_vector: Tensor of shape (m,) — the user's rating row
                           (may include perturbation δ that carries grad).

        Returns:
            Tensor of shape (m,) of predicted scores for all items.
        """
        V = self.item_factors  # (m, d)

        # Soft weights: sigmoid centered at 0.5 so unrated items (r=0) get
        # near-zero weight while new ratings smoothly gain influence.
        # sigmoid((0 - 0.5)*20) ≈ 0.00005, sigmoid((1 - 0.5)*20) ≈ 0.99995
        weights = torch.sigmoid((rating_vector - 0.5) * 20.0)  # (m,)

        if weights.sum() < 1e-10:
            return self.item_bias + self.global_bias

        r_centered = rating_vector - self.item_bias - self.global_bias  # (m,)

        # Weighted ridge regression: u* = (V^T W V + λI)^{-1} V^T (w ⊙ r)
        wV = weights.unsqueeze(1) * V  # (m, d) — each row scaled by weight
        lam = self.weight_decay * weights.sum()
        A = wV.T @ V + lam * torch.eye(self.n_factors)
        b = V.T @ (weights * r_centered)
        u_star = torch.linalg.solve(A, b)  # (d,)

        scores = u_star @ V.T + self.item_bias + self.global_bias
        return scores

    # ------------------------------------------------------------------
    # Retrain (cross-user coupling)
    # ------------------------------------------------------------------

    def _retrain_and_score(self, user_id: int, rating_matrix: np.ndarray) -> np.ndarray:
        """Warm-start retrain on modified matrix and score via ridge regression.

        Called when other users' rows changed. Retrains U, V, bu, bi from
        the base state for a limited number of SGD steps, then uses the
        retrained item factors to re-solve the victim's user embedding.
        """
        users, items = np.nonzero(rating_matrix)
        ratings = rating_matrix[users, items].astype(np.float32)
        gb = float(ratings.mean())

        ds = TensorDataset(
            torch.from_numpy(users.astype(np.int64)),
            torch.from_numpy(items.astype(np.int64)),
            torch.from_numpy(ratings - gb),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        # Initialise from base state (warm start)
        U = nn.Embedding(self.n_users, self.n_factors)
        V = nn.Embedding(self.n_items, self.n_factors)
        bu = nn.Embedding(self.n_users, 1)
        bi = nn.Embedding(self.n_items, 1)
        U.weight.data = self._base_state["user_factors"].clone()
        V.weight.data = self._base_state["item_factors"].clone()
        bu.weight.data = self._base_state["user_bias"].clone().unsqueeze(1)
        bi.weight.data = self._base_state["item_bias"].clone().unsqueeze(1)

        opt = torch.optim.Adam(
            list(U.parameters()) + list(V.parameters()) +
            list(bu.parameters()) + list(bi.parameters()),
            lr=self.lr * 0.5, weight_decay=self.weight_decay,
        )

        step = 0
        for _ in range(max(1, self.retrain_steps // max(len(loader), 1) + 1)):
            for u_batch, i_batch, r_batch in loader:
                pred = (U(u_batch) * V(i_batch)).sum(dim=1) + bu(u_batch).squeeze() + bi(i_batch).squeeze()
                loss = nn.functional.mse_loss(pred, r_batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                step += 1
                if step >= self.retrain_steps:
                    break
            if step >= self.retrain_steps:
                break

        # Score user_id via ridge regression with retrained item factors
        V_new = V.weight.detach()
        bi_new = bi.weight.detach().squeeze()

        r = torch.tensor(rating_matrix[user_id], dtype=torch.float32)
        weights = torch.sigmoid((r - 0.5) * 20.0)

        if weights.sum() < 1e-10:
            return (bi_new + gb).numpy()

        r_centered = r - bi_new - gb
        wV = weights.unsqueeze(1) * V_new
        lam = self.weight_decay * weights.sum()
        A = wV.T @ V_new + lam * torch.eye(self.n_factors)
        b = V_new.T @ (weights * r_centered)
        u_star = torch.linalg.solve(A, b)

        scores = u_star @ V_new.T + bi_new + gb
        return scores.detach().numpy()

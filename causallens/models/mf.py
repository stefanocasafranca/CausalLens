"""Matrix Factorization recommender — implemented from scratch with PyTorch."""

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
    """

    def __init__(self, n_users: int, n_items: int, n_factors: int = 64,
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 n_epochs: int = 20, batch_size: int = 1024,
                 device: str | None = None):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Learnable parameters
        self.user_factors: torch.Tensor | None = None
        self.item_factors: torch.Tensor | None = None
        self.user_bias: torch.Tensor | None = None
        self.item_bias: torch.Tensor | None = None
        self.global_bias: float = 0.0

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
        return self

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def get_scores(self, user_id: int, rating_matrix: np.ndarray) -> np.ndarray:
        """Predict scores for all items.

        Re-solves for the user embedding via ridge regression on the
        observed entries of rating_matrix[user_id], keeping item factors
        fixed. This means perturbed rating matrices produce different
        scores — essential for both whitebox and blackbox metrics.
        """
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

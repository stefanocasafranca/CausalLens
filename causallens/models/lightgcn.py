"""LightGCN recommender — implemented from scratch with PyTorch.

Light Graph Convolution Network (He et al., SIGIR 2020).
Multi-layer graph convolution on the user-item bipartite graph with no
feature transformation or nonlinear activation.  Final embeddings are
the mean of all layer embeddings (layer 0 through layer L).

Cross-user coupling: item embeddings propagate through the bipartite
graph, so retraining on perturbed data shifts all users' scores.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
import copy

from causallens.recommender import Recommender


# ------------------------------------------------------------------
# Adjacency helpers
# ------------------------------------------------------------------

def _build_normalized_adj(rating_matrix: np.ndarray) -> torch.Tensor:
    """Build normalized bipartite adjacency D^{-1/2} A D^{-1/2}.

    Node layout: [user_0 .. user_{n-1}, item_0 .. item_{m-1}].
    Edges are bidirectional between users and items with nonzero ratings.
    """
    n_u, n_i = rating_matrix.shape
    users, items = np.nonzero(rating_matrix)
    n = n_u + n_i

    # Bidirectional edges: user→item and item→user
    row = np.concatenate([users, items + n_u])
    col = np.concatenate([items + n_u, users])

    # Degree normalization
    degree = np.zeros(n, dtype=np.float32)
    np.add.at(degree, row, 1.0)
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    vals = (d_inv_sqrt[row] * d_inv_sqrt[col]).astype(np.float32)

    indices = torch.tensor(np.stack([row, col]), dtype=torch.long)
    values = torch.tensor(vals, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


# ------------------------------------------------------------------
# Module
# ------------------------------------------------------------------

class LightGCNModule(nn.Module):
    """PyTorch module for LightGCN."""

    def __init__(self, n_users: int, n_items: int,
                 embed_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers

        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)

        nn.init.normal_(self.user_embedding.weight, 0, 0.01)
        nn.init.normal_(self.item_embedding.weight, 0, 0.01)

    def forward(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """GCN propagation with mean pooling across layers.

        Returns (user_embeddings, item_embeddings).
        """
        all_emb = torch.cat([self.user_embedding.weight,
                             self.item_embedding.weight], dim=0)
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(adj, all_emb)
            embs.append(all_emb)

        final = torch.stack(embs, dim=0).mean(dim=0)
        return final[:self.n_users], final[self.n_users:]


# ------------------------------------------------------------------
# Recommender wrapper
# ------------------------------------------------------------------

class LightGCN(Recommender):
    """LightGCN recommender with retraining support for causal metrics.

    get_scores behaviour:
    - Same R as training  → cached GCN forward (fast).
    - Only user's row changed  → rebuild adjacency + forward (no retrain).
    - Other rows changed  → warm-start retrain (cross-user coupling).
    """

    def __init__(self, n_users: int, n_items: int, embed_dim: int = 64,
                 n_layers: int = 3, lr: float = 1e-3,
                 weight_decay: float = 1e-4, n_epochs: int = 20,
                 batch_size: int = 2048, retrain_steps: int = 25):
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.retrain_steps = retrain_steps

        self.model: LightGCNModule | None = None
        self._adj: torch.Tensor | None = None
        self._training_matrix: np.ndarray | None = None
        self._base_state: dict | None = None
        self._rating_mean: float = 0.0

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    def fit(self, rating_matrix: np.ndarray,
            verbose: bool = True) -> "LightGCN":
        """Train LightGCN on observed entries via MSE."""
        self._training_matrix = rating_matrix.copy()

        self.model = LightGCNModule(self.n_users, self.n_items,
                                    self.embed_dim, self.n_layers)
        self._adj = _build_normalized_adj(rating_matrix)

        users, items = np.nonzero(rating_matrix)
        ratings = rating_matrix[users, items].astype(np.float32)
        self._rating_mean = float(ratings.mean())

        ds = TensorDataset(
            torch.from_numpy(users.astype(np.int64)),
            torch.from_numpy(items.astype(np.int64)),
            torch.from_numpy(ratings - self._rating_mean),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)

        rng = trange(self.n_epochs, desc="LightGCN training",
                     disable=not verbose)
        for _ in rng:
            total_loss = 0.0
            for u_batch, i_batch, r_batch in loader:
                user_emb, item_emb = self.model.forward(self._adj)
                pred = (user_emb[u_batch] * item_emb[i_batch]).sum(dim=1)
                loss = nn.functional.mse_loss(pred, r_batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(r_batch)
            rng.set_postfix(mse=total_loss / len(ratings))

        self._base_state = copy.deepcopy(self.model.state_dict())
        return self

    # ----------------------------------------------------------
    # Scoring
    # ----------------------------------------------------------

    def get_scores(self, user_id: int,
                   rating_matrix: np.ndarray) -> np.ndarray:
        diff = (rating_matrix != self._training_matrix)
        changed_rows = np.where(diff.any(axis=1))[0]

        if len(changed_rows) == 0:
            # No change — cached forward
            with torch.no_grad():
                u_emb, i_emb = self.model.forward(self._adj)
                scores = (u_emb[user_id] @ i_emb.T).numpy()
            return scores + self._rating_mean

        only_self = (len(changed_rows) == 1 and changed_rows[0] == user_id)

        if only_self:
            # Self-change: rebuild adjacency, forward pass (no retrain)
            adj = _build_normalized_adj(rating_matrix)
            with torch.no_grad():
                u_emb, i_emb = self.model.forward(adj)
                scores = (u_emb[user_id] @ i_emb.T).numpy()
            return scores + self._rating_mean

        # Cross-user change — warm-start retrain
        return self._retrain_and_score(user_id, rating_matrix)

    # ----------------------------------------------------------
    # Retrain (cross-user coupling)
    # ----------------------------------------------------------

    def _retrain_and_score(self, user_id: int,
                           rating_matrix: np.ndarray) -> np.ndarray:
        """Warm-start retrain on modified data, then score."""
        model = LightGCNModule(self.n_users, self.n_items,
                               self.embed_dim, self.n_layers)
        model.load_state_dict(copy.deepcopy(self._base_state))
        model.train()

        adj = _build_normalized_adj(rating_matrix)

        users, items = np.nonzero(rating_matrix)
        ratings = rating_matrix[users, items].astype(np.float32)
        rm = float(ratings.mean())

        ds = TensorDataset(
            torch.from_numpy(users.astype(np.int64)),
            torch.from_numpy(items.astype(np.int64)),
            torch.from_numpy(ratings - rm),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.Adam(model.parameters(), lr=self.lr * 0.5,
                               weight_decay=self.weight_decay)

        step = 0
        for _ in range(max(1, self.retrain_steps // max(len(loader), 1) + 1)):
            for u_batch, i_batch, r_batch in loader:
                u_emb, i_emb = model.forward(adj)
                pred = (u_emb[u_batch] * i_emb[i_batch]).sum(dim=1)
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
        with torch.no_grad():
            u_emb, i_emb = model.forward(adj)
            scores = (u_emb[user_id] @ i_emb.T).numpy()
        return scores + rm

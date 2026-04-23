"""SASRec recommender — implemented from scratch with PyTorch.

Self-Attentive Sequential Recommendation (Kang & McAuley, ICDM 2018).
Transformer encoder on user interaction sequences with causal masking.
Items are ordered by item-id within each user (deterministic proxy for
temporal order when timestamps are unavailable).

Cross-user coupling: item embeddings and transformer weights are shared,
so retraining on perturbed data shifts all users' scores.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
import copy

from causallens.recommender import Recommender


# ------------------------------------------------------------------
# Module
# ------------------------------------------------------------------

class SASRecModule(nn.Module):
    """Transformer-based sequential recommender."""

    def __init__(self, n_items: int, embed_dim: int = 64,
                 n_heads: int = 2, n_layers: int = 2,
                 max_seq_len: int = 50, dropout: float = 0.2):
        super().__init__()
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Item embeddings: 0 = padding, items are 1-indexed internally
        self.item_embedding = nn.Embedding(n_items + 1, embed_dim,
                                           padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout_layer = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Encode interaction sequences.

        Args:
            seq: (batch, seq_len) item IDs, 1-indexed, left-padded with 0.

        Returns:
            (batch, seq_len, embed_dim) hidden states.
        """
        seq_len = seq.size(1)
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0)

        x = self.item_embedding(seq) + self.position_embedding(positions)
        x = self.layer_norm(self.dropout_layer(x))

        # Causal mask — attend only to current and past positions
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=seq.device), diagonal=1
        ).bool()
        padding_mask = (seq == 0)

        x = self.transformer(x, mask=causal_mask,
                             src_key_padding_mask=padding_mask)
        return x

    def score_all_items(self, seq: torch.Tensor) -> torch.Tensor:
        """Score all items using last-position output.

        Returns:
            (batch, n_items) scores (0-indexed item space).
        """
        x = self.forward(seq)                       # (B, L, d)
        last_hidden = x[:, -1, :]                   # (B, d)
        item_emb = self.item_embedding.weight[1:]   # (n_items, d)
        return last_hidden @ item_emb.T             # (B, n_items)


# ------------------------------------------------------------------
# Sequence helpers
# ------------------------------------------------------------------

def _build_sequences(rating_matrix: np.ndarray,
                     max_seq_len: int) -> np.ndarray:
    """Build left-padded, 1-indexed interaction sequences for all users.

    Items are sorted by item-id (deterministic ordering).
    """
    n_u = rating_matrix.shape[0]
    sequences = np.zeros((n_u, max_seq_len), dtype=np.int64)
    for u in range(n_u):
        rated = np.sort(np.where(rating_matrix[u] > 0)[0])
        if len(rated) > max_seq_len:
            rated = rated[-max_seq_len:]
        if len(rated) > 0:
            sequences[u, -len(rated):] = rated + 1   # 1-indexed
    return sequences


def _build_user_sequence(user_ratings: np.ndarray,
                         max_seq_len: int) -> np.ndarray:
    """Build sequence for a single user (1-D rating vector)."""
    seq = np.zeros(max_seq_len, dtype=np.int64)
    rated = np.sort(np.where(user_ratings > 0)[0])
    if len(rated) > max_seq_len:
        rated = rated[-max_seq_len:]
    if len(rated) > 0:
        seq[-len(rated):] = rated + 1
    return seq


# ------------------------------------------------------------------
# Recommender wrapper
# ------------------------------------------------------------------

class SASRec(Recommender):
    """SASRec recommender with retraining support for causal metrics.

    get_scores behaviour:
    - Same R as training  → cached transformer forward.
    - Only user's row changed  → rebuild user sequence + forward (no retrain).
    - Other rows changed  → warm-start retrain (cross-user coupling).
    """

    def __init__(self, n_users: int, n_items: int,
                 embed_dim: int = 64, n_heads: int = 2,
                 n_layers: int = 2, max_seq_len: int = 50,
                 dropout: float = 0.2, lr: float = 1e-3,
                 weight_decay: float = 1e-5, n_epochs: int = 30,
                 batch_size: int = 256, retrain_steps: int = 25):
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.retrain_steps = retrain_steps

        self.model: SASRecModule | None = None
        self._training_matrix: np.ndarray | None = None
        self._base_state: dict | None = None
        self._sequences: np.ndarray | None = None
        self._cached_scores: dict[int, np.ndarray] = {}  # uid → scores

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    def fit(self, rating_matrix: np.ndarray,
            verbose: bool = True) -> "SASRec":
        """Train SASRec on interaction sequences."""
        self._training_matrix = rating_matrix.copy()
        self._sequences = _build_sequences(rating_matrix, self.max_seq_len)

        self.model = SASRecModule(
            self.n_items, self.embed_dim, self.n_heads,
            self.n_layers, self.max_seq_len, self.dropout,
        )

        self._train_model(self.model, self._sequences,
                          self.n_epochs, self.lr, verbose=verbose)

        self._base_state = copy.deepcopy(self.model.state_dict())
        return self

    def _train_model(self, model: SASRecModule,
                     sequences: np.ndarray, n_epochs: int,
                     lr: float, verbose: bool = True,
                     max_steps: int | None = None):
        """Train with next-item prediction (cross-entropy)."""
        model.train()

        # Need at least 2 items per sequence for input→target
        valid_mask = (sequences != 0).sum(axis=1) >= 2
        valid_seqs = sequences[valid_mask]
        if len(valid_seqs) == 0:
            return

        input_seqs = torch.from_numpy(valid_seqs[:, :-1])
        target_seqs = torch.from_numpy(valid_seqs[:, 1:])
        n_train = len(valid_seqs)

        opt = torch.optim.Adam(model.parameters(), lr=lr,
                               weight_decay=self.weight_decay)

        step = 0
        rng = trange(n_epochs, desc="SASRec training", disable=not verbose)
        for _ in rng:
            perm = np.random.permutation(n_train)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, self.batch_size):
                end = min(start + self.batch_size, n_train)
                idx = perm[start:end]

                inp = input_seqs[idx]
                tgt = target_seqs[idx]

                output = model.forward(inp)   # (B, L-1, d)
                item_emb = model.item_embedding.weight[1:]  # (n_items, d)
                logits = output @ item_emb.T  # (B, L-1, n_items)

                tgt_flat = tgt.reshape(-1)
                logits_flat = logits.reshape(-1, model.n_items)

                valid = tgt_flat > 0
                if valid.sum() == 0:
                    continue

                loss = nn.functional.cross_entropy(
                    logits_flat[valid], tgt_flat[valid] - 1  # 0-indexed
                )

                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches += 1
                step += 1

                if max_steps is not None and step >= max_steps:
                    break

            if n_batches > 0:
                rng.set_postfix(ce_loss=total_loss / n_batches)
            if max_steps is not None and step >= max_steps:
                break

        model.eval()

    # ----------------------------------------------------------
    # Scoring
    # ----------------------------------------------------------

    def get_scores(self, user_id: int,
                   rating_matrix: np.ndarray) -> np.ndarray:
        # Fast path: check user's row first
        user_same = np.array_equal(
            rating_matrix[user_id], self._training_matrix[user_id])

        if user_same:
            if np.array_equal(rating_matrix, self._training_matrix):
                # No change — use cache
                if user_id not in self._cached_scores:
                    seq = torch.from_numpy(self._sequences[user_id:user_id + 1])
                    with torch.no_grad():
                        scores = self.model.score_all_items(seq)
                    self._cached_scores[user_id] = scores[0].numpy()
                return self._cached_scores[user_id].copy()
            # Cross-user change
            return self._retrain_and_score(user_id, rating_matrix)

        # User's row changed — check if only self
        diff = (rating_matrix != self._training_matrix)
        changed_rows = np.where(diff.any(axis=1))[0]

        if len(changed_rows) == 1:
            # Self-change: rebuild sequence + forward (fast)
            user_seq = _build_user_sequence(rating_matrix[user_id],
                                            self.max_seq_len)
            seq = torch.from_numpy(user_seq[np.newaxis, :])
            with torch.no_grad():
                scores = self.model.score_all_items(seq)
            return scores[0].numpy()

        # Cross-user change — warm-start retrain
        return self._retrain_and_score(user_id, rating_matrix)

    # ----------------------------------------------------------
    # Retrain (cross-user coupling)
    # ----------------------------------------------------------

    def _retrain_and_score(self, user_id: int,
                           rating_matrix: np.ndarray) -> np.ndarray:
        """Warm-start retrain on modified data, then score."""
        model = SASRecModule(
            self.n_items, self.embed_dim, self.n_heads,
            self.n_layers, self.max_seq_len, self.dropout,
        )
        model.load_state_dict(copy.deepcopy(self._base_state))

        # Incrementally update only changed sequences
        sequences = self._sequences.copy()
        diff = (rating_matrix != self._training_matrix)
        for u in np.where(diff.any(axis=1))[0]:
            sequences[u] = _build_user_sequence(rating_matrix[u],
                                                self.max_seq_len)

        self._train_model(model, sequences, n_epochs=100,
                          lr=self.lr * 0.5, verbose=False,
                          max_steps=self.retrain_steps)

        user_seq = _build_user_sequence(rating_matrix[user_id],
                                        self.max_seq_len)
        seq = torch.from_numpy(user_seq[np.newaxis, :])
        with torch.no_grad():
            scores = model.score_all_items(seq)
        return scores[0].numpy()

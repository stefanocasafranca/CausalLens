"""Abstract Recommender interface for CausalLens."""

from abc import ABC, abstractmethod
import numpy as np


class Recommender(ABC):
    """Abstract base class for recommender systems.

    All recommenders must implement these three methods to be auditable
    by CausalLens.
    """

    @abstractmethod
    def get_scores(self, user_id: int, rating_matrix: np.ndarray) -> np.ndarray:
        """Return predicted scores for all items given a rating matrix.

        Args:
            user_id: Index of the user in the rating matrix.
            rating_matrix: Full n×m rating matrix (may be perturbed).

        Returns:
            1-D array of shape (m,) with predicted scores for each item.
        """

    def get_recommendations(self, user_id: int, rating_matrix: np.ndarray, k: int = 10) -> np.ndarray:
        """Return top-k recommended item indices.

        Args:
            user_id: Index of the user.
            rating_matrix: Full n×m rating matrix.
            k: Number of items to recommend.

        Returns:
            1-D array of k item indices, sorted by descending score.
        """
        scores = self.get_scores(user_id, rating_matrix)
        # Mask items the user has already rated
        rated_mask = rating_matrix[user_id] > 0
        scores[rated_mask] = -np.inf
        return np.argsort(scores)[::-1][:k]

    def submit_feedback(self, user_id: int, item_id: int, rating: float,
                        rating_matrix: np.ndarray) -> np.ndarray:
        """Record a new rating and return the updated matrix.

        Default implementation just writes into the matrix. Subclasses that
        need retraining can override.
        """
        rating_matrix = rating_matrix.copy()
        rating_matrix[user_id, item_id] = rating
        return rating_matrix

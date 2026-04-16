"""MovieLens-1M data loader."""

import os
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DEFAULT_DATA_DIR = Path("data")


def load_movielens_1m(data_dir: str | Path | None = None,
                      min_user_ratings: int = 20,
                      min_item_ratings: int = 5) -> dict:
    """Download (if needed) and load MovieLens-1M as a rating matrix.

    Returns:
        dict with keys:
            rating_matrix: np.ndarray of shape (n_users, n_items), float32
            user_ids: original user IDs corresponding to matrix rows
            item_ids: original item IDs corresponding to matrix columns
            user_map: dict original_id -> matrix_row
            item_map: dict original_id -> matrix_col
            df: the filtered pandas DataFrame
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    ml_dir = data_dir / "ml-1m"
    ratings_path = ml_dir / "ratings.dat"

    # Download and extract if not present
    if not ratings_path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        zip_path = data_dir / "ml-1m.zip"
        if not zip_path.exists():
            print(f"Downloading MovieLens-1M to {zip_path}...")
            urllib.request.urlretrieve(ML1M_URL, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)

    # Parse the :: separated file
    df = pd.read_csv(
        ratings_path,
        sep="::",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )

    # Filter sparse users/items
    while True:
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        valid_items = item_counts[item_counts >= min_item_ratings].index
        filtered = df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]
        if len(filtered) == len(df):
            break
        df = filtered

    # Create contiguous ID mappings
    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["item_id"].unique())
    user_map = {uid: idx for idx, uid in enumerate(unique_users)}
    item_map = {iid: idx for idx, iid in enumerate(unique_items)}

    n_users = len(unique_users)
    n_items = len(unique_items)

    # Build rating matrix
    rating_matrix = np.zeros((n_users, n_items), dtype=np.float32)
    for _, row in df.iterrows():
        rating_matrix[user_map[row["user_id"]], item_map[row["item_id"]]] = row["rating"]

    print(f"MovieLens-1M loaded: {n_users} users, {n_items} items, "
          f"{(rating_matrix > 0).sum()} ratings, "
          f"density {(rating_matrix > 0).mean():.4f}")

    return {
        "rating_matrix": rating_matrix,
        "user_ids": np.array(unique_users),
        "item_ids": np.array(unique_items),
        "user_map": user_map,
        "item_map": item_map,
        "df": df,
    }

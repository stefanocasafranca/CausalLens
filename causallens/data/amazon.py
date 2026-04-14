"""Amazon Digital Music data loader (5-core ratings)."""

import gzip
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


# Amazon Reviews v2 — 5-core Digital Music ratings
AMAZON_DM_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Digital_Music_5.json.gz"
DEFAULT_DATA_DIR = Path("data")


def load_amazon_digital_music(data_dir: str | Path | None = None,
                              min_user_ratings: int = 10,
                              min_item_ratings: int = 5) -> dict:
    """Download (if needed) and load Amazon Digital Music 5-core ratings.

    Returns:
        dict with keys:
            rating_matrix: np.ndarray of shape (n_users, n_items), float32
            user_ids: original user IDs (strings) corresponding to matrix rows
            item_ids: original item IDs (ASINs) corresponding to matrix columns
            user_map: dict original_id -> matrix_row
            item_map: dict original_id -> matrix_col
            df: the filtered pandas DataFrame
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    gz_path = data_dir / "Digital_Music_5.json.gz"

    # Download if not present
    if not gz_path.exists():
        print(f"Downloading Amazon Digital Music to {gz_path}...")
        urllib.request.urlretrieve(AMAZON_DM_URL, gz_path)

    # Parse JSONL (one JSON object per line, gzipped)
    print("Parsing Amazon Digital Music ratings...")
    records = []
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            if "overall" in obj and "reviewerID" in obj and "asin" in obj:
                records.append({
                    "user_id": obj["reviewerID"],
                    "item_id": obj["asin"],
                    "rating": float(obj["overall"]),
                    "timestamp": int(obj.get("unixReviewTime", 0)),
                })

    df = pd.DataFrame(records)
    print(f"  Raw: {len(df)} ratings, {df['user_id'].nunique()} users, {df['item_id'].nunique()} items")

    # Filter sparse users/items iteratively
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

    print(f"Amazon Digital Music loaded: {n_users} users, {n_items} items, "
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

"""Amazon Digital Music data loader (5-core ratings)."""

import gzip
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path("data")

# Amazon Musical Instruments 5-core (2023, CSV from HuggingFace)
_MI_CSV_CANDIDATES = [
    "data/amazon_2023/benchmark/5core/rating_only/Musical_Instruments.csv",
]
_MI_HF_PATH = "benchmark/5core/rating_only/Musical_Instruments.csv"

# Legacy Digital Music JSONL candidates
_DM_JSONL_CANDIDATES = [
    "data/amazon_2023/raw/review_categories/Digital_Music.jsonl",
    "data/Digital_Music_5.json.gz",
]


def load_amazon_digital_music(data_dir: str | Path | None = None,
                              min_user_ratings: int = 10,
                              min_item_ratings: int = 5) -> dict:
    """Load Amazon Musical Instruments 5-core ratings (2023).

    Falls back to Digital Music JSONL if available.

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

    # Try 5-core CSV first (Musical Instruments)
    csv_path = None
    for cand in _MI_CSV_CANDIDATES:
        if Path(cand).exists():
            csv_path = Path(cand)
            break

    if csv_path is not None:
        return _load_from_csv(csv_path, min_user_ratings, min_item_ratings)

    # Try JSONL candidates (Digital Music legacy)
    jsonl_path = None
    for cand in _DM_JSONL_CANDIDATES:
        if Path(cand).exists():
            jsonl_path = Path(cand)
            break

    if jsonl_path is not None:
        return _load_from_jsonl(jsonl_path, min_user_ratings, min_item_ratings)

    # Download from HuggingFace
    try:
        from huggingface_hub import hf_hub_download
        csv_path = Path(hf_hub_download(
            repo_id="McAuley-Lab/Amazon-Reviews-2023",
            filename=_MI_HF_PATH,
            repo_type="dataset",
            local_dir=str(data_dir / "amazon_2023"),
        ))
        print(f"Downloaded Amazon Musical Instruments from HuggingFace to {csv_path}")
        return _load_from_csv(csv_path, min_user_ratings, min_item_ratings)
    except Exception as e:
        raise FileNotFoundError(
            f"Cannot find or download Amazon dataset. Error: {e}"
        )


def _load_from_csv(csv_path, min_user_ratings, min_item_ratings):
    """Load from 5-core CSV (columns: user_id, parent_asin, rating, timestamp)."""
    print(f"Loading Amazon Musical Instruments from {csv_path}...")
    df = pd.read_csv(csv_path)
    df.columns = ["user_id", "item_id", "rating", "timestamp"]
    print(f"  Raw: {len(df)} ratings, {df['user_id'].nunique()} users, {df['item_id'].nunique()} items")

    return _build_matrix(df, min_user_ratings, min_item_ratings, "Amazon Musical Instruments")


def _load_from_jsonl(jsonl_path, min_user_ratings, min_item_ratings):
    """Load from JSONL (v2 or 2023 format)."""
    print(f"Parsing Amazon Digital Music from {jsonl_path}...")
    records = []
    opener = gzip.open if str(jsonl_path).endswith(".gz") else open
    with opener(jsonl_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "reviewerID" in obj:
                uid = obj["reviewerID"]
                rating = float(obj.get("overall", 0))
            elif "user_id" in obj:
                uid = obj["user_id"]
                rating = float(obj.get("rating", 0))
            else:
                continue
            if rating > 0 and "asin" in obj:
                records.append({"user_id": uid, "item_id": obj["asin"], "rating": rating})

    df = pd.DataFrame(records)
    print(f"  Raw: {len(df)} ratings, {df['user_id'].nunique()} users, {df['item_id'].nunique()} items")
    return _build_matrix(df, min_user_ratings, min_item_ratings, "Amazon Digital Music")


def _build_matrix(df, min_user_ratings, min_item_ratings, name):
    """Filter and build rating matrix from DataFrame."""
    while True:
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        valid_items = item_counts[item_counts >= min_item_ratings].index
        filtered = df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]
        if len(filtered) == len(df):
            break
        df = filtered

    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["item_id"].unique())
    user_map = {uid: idx for idx, uid in enumerate(unique_users)}
    item_map = {iid: idx for idx, iid in enumerate(unique_items)}

    n_users = len(unique_users)
    n_items = len(unique_items)

    rating_matrix = np.zeros((n_users, n_items), dtype=np.float32)
    for _, row in df.iterrows():
        rating_matrix[user_map[row["user_id"]], item_map[row["item_id"]]] = row["rating"]

    print(f"{name} loaded: {n_users} users, {n_items} items, "
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

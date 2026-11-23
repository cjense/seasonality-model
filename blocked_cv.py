"""
Blocked cross-validation folds shared by select_features.py and run_xgboost.py.

The period is cut into chunks of CV_BLOCK_YEARS calendar years. Each fold holds out one chunk
and trains on every other row except those within CV_GAP_MONTHS of the held-out chunk, so the
model is trained on years both before and after the one it is tested on.
"""
import numpy as np
import pandas as pd

from config import CV_BLOCK_YEARS, CV_GAP_MONTHS


def chunk_labels(times: pd.Series) -> np.ndarray:
    """Chunk index of each row, counted in CV_BLOCK_YEARS-year steps from the first year present."""
    years = pd.to_datetime(times).dt.year.to_numpy()
    return (years - years.min()) // CV_BLOCK_YEARS


def blocked_splits(times: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    (train_positions, test_positions) for each chunk, as row positions into `times`.
    Works for one row per timestep or many (pixels): all rows of a timestep share a fold.
    A plain list, so sklearn's cross_val_score and SequentialFeatureSelector accept it as `cv`.
    """
    t = pd.to_datetime(times).reset_index(drop=True)
    chunks = chunk_labels(t)
    gap = pd.DateOffset(months=CV_GAP_MONTHS)
    splits = []
    for c in np.unique(chunks):
        held_out = chunks == c
        start, end = t[held_out].min(), t[held_out].max()
        near = t.between(start - gap, end + gap).to_numpy()
        splits.append((np.flatnonzero(~near), np.flatnonzero(held_out)))
    return splits


def chunk_names(times: pd.Series) -> list[str]:
    """Readable name of each chunk, in the same order as blocked_splits()."""
    years = pd.to_datetime(times).dt.year
    chunks = chunk_labels(times)
    return [f"{years[chunks == c].min()}" if CV_BLOCK_YEARS == 1 else f"{years[chunks == c].min()}–{years[chunks == c].max()}"
            for c in np.unique(chunks)]

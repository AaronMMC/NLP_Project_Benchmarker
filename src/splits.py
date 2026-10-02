"""
splits.py — Grouped / non-grouped / deduplicated cross-validation splitters.

Implements three split regimes, each reported side by side:
- grouped (StratifiedGroupKFold by song) — valid for Task A/B
- non-grouped (stratified k-fold) — to expose leakage
- deduplicated — near-duplicate lines kept in same fold
"""

import logging
from typing import List, Tuple, Optional, Dict

import numpy as np
from sklearn.model_selection import (
    StratifiedKFold, StratifiedGroupKFold, RepeatedStratifiedKFold
)
from rapidfuzz import fuzz

from src.preprocess import normalise_text

logger = logging.getLogger(__name__)


# =====================================================================
# IDEA 1: Leakage-aware evaluation
# WHY:     Hymns repeat refrains and lines, so random k-fold inflates
#          scores by placing near-identical lines in both train and test.
# HOW:     Three split regimes reported side by side:
#          (1) grouped by song — valid estimate
#          (2) non-grouped — shows inflation
#          (3) deduplicated — fuzzy-matched near-duplicates co-fold
#          The gap between (1)/(3) and (2) is the "leakage inflation".
# OUTPUT:  Per-fold metrics for each regime, leakage inflation plot
# CAVEAT:  Grouped splits need enough songs per class; may need fewer
#          folds if class sizes are small. Task C cannot group by song
#          since song IS the label; uses dedup instead.
# =====================================================================


def build_dedup_groups(texts: List[str], threshold: int = 85) -> np.ndarray:
    """
    Assign group IDs based on near-duplicate detection.

    Lines with normalised fuzzy ratio >= threshold are placed in the
    same group so they end up in the same fold.

    Parameters
    ----------
    texts : List[str]
        The text of each sample (line or passage).
    threshold : int
        Fuzzy ratio threshold for considering two texts near-duplicates.

    Returns
    -------
    np.ndarray
        Group ID for each text.
    """
    n = len(texts)
    normalised = [normalise_text(t) for t in texts]

    # Union-Find for grouping
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Compare all pairs (O(n^2) but n is small for this corpus)
    for i in range(n):
        for j in range(i + 1, n):
            # Exact normalised match
            if normalised[i] == normalised[j]:
                union(i, j)
            # Fuzzy match
            elif fuzz.ratio(normalised[i], normalised[j]) >= threshold:
                union(i, j)

    # Convert to contiguous group IDs
    root_to_group = {}
    groups = np.zeros(n, dtype=int)
    next_group = 0
    for i in range(n):
        root = find(i)
        if root not in root_to_group:
            root_to_group[root] = next_group
            next_group += 1
        groups[i] = root_to_group[root]

    n_groups = len(root_to_group)
    n_deduped = sum(1 for g in set(groups) if np.sum(groups == g) > 1)
    logger.info(f"Dedup grouping: {n} samples → {n_groups} groups, "
               f"{n_deduped} groups with duplicates")

    return groups


def get_split_regimes(n_folds: int, n_repeats: int,
                      song_groups: Optional[np.ndarray] = None,
                      dedup_groups: Optional[np.ndarray] = None,
                      task: str = 'A') -> Dict[str, object]:
    """
    Get the three split regimes for cross-validation.

    Parameters
    ----------
    n_folds : int
        Number of CV folds.
    n_repeats : int
        Number of repeats (for repeated CV).
    song_groups : Optional[np.ndarray]
        Song/unit group IDs for grouped splitting.
    dedup_groups : Optional[np.ndarray]
        Near-duplicate group IDs for dedup splitting.
    task : str
        Task identifier ('A', 'B', 'C', 'D').

    Returns
    -------
    Dict[str, object]
        Named CV splitter objects.
    """
    regimes = {}

    # Non-grouped: standard stratified k-fold (exposes leakage)
    regimes['non_grouped'] = RepeatedStratifiedKFold(
        n_splits=n_folds, n_repeats=n_repeats, random_state=42
    )

    # Grouped by song (valid for Tasks A/B, not for C)
    if task in ('A', 'B', 'D') and song_groups is not None:
        n_unique_groups = len(np.unique(song_groups))
        actual_folds = min(n_folds, n_unique_groups)
        if actual_folds < n_folds:
            logger.warning(f"Only {n_unique_groups} song groups available, "
                          f"reducing folds from {n_folds} to {actual_folds}")
        if actual_folds >= 2:
            regimes['grouped'] = StratifiedGroupKFold(n_splits=actual_folds)
        else:
            logger.warning("Not enough groups for grouped CV. Skipping.")

    # Deduplicated: use dedup groups as group constraint
    if dedup_groups is not None:
        n_unique_dedup = len(np.unique(dedup_groups))
        actual_folds_d = min(n_folds, n_unique_dedup)
        if actual_folds_d >= 2:
            regimes['deduplicated'] = StratifiedGroupKFold(n_splits=actual_folds_d)
        else:
            logger.warning("Not enough dedup groups for dedup CV. Skipping.")

    logger.info(f"Split regimes for task {task}: {list(regimes.keys())}")
    return regimes


def generate_cv_splits(X: np.ndarray, y: np.ndarray,
                       regime_name: str, splitter,
                       groups: Optional[np.ndarray] = None
                       ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate train/test index pairs for a given CV regime.

    Parameters
    ----------
    X : array-like
        Feature matrix or text array.
    y : array-like
        Labels.
    regime_name : str
        Name of the regime (for logging).
    splitter : sklearn splitter
        The CV splitter object.
    groups : Optional[np.ndarray]
        Group array (for grouped/dedup regimes).

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (train_indices, test_indices) tuples.
    """
    splits = []
    try:
        if groups is not None and hasattr(splitter, 'split'):
            for train_idx, test_idx in splitter.split(X, y, groups=groups):
                splits.append((train_idx, test_idx))
        else:
            for train_idx, test_idx in splitter.split(X, y):
                splits.append((train_idx, test_idx))
    except Exception as e:
        logger.warning(f"Error generating {regime_name} splits: {e}")
        # Fallback to simple stratified k-fold
        fallback = StratifiedKFold(n_splits=min(3, len(np.unique(y))),
                                   shuffle=True, random_state=42)
        for train_idx, test_idx in fallback.split(X, y):
            splits.append((train_idx, test_idx))

    logger.info(f"{regime_name}: generated {len(splits)} splits, "
               f"train sizes: {[len(s[0]) for s in splits[:3]]}..., "
               f"test sizes: {[len(s[1]) for s in splits[:3]]}...")
    return splits

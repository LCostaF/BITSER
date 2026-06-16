"""
split_manager.py
----------------
Central module for reproducible stratified train/test splitting.

Responsibilities
~~~~~~~~~~~~~~~~
* Generate or validate a random seed.
* Produce a stratified 80/20 split from a full metadata DataFrame.
* Reconstruct the exact same split from a persisted seed + split definition.
* Guarantee zero leakage: test indices are never touched during CV or training.
"""

import random
import uuid
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def generate_seed() -> int:
    """Return a fresh random integer seed in [0, 2**31 - 1]."""
    return random.randint(0, 2**31 - 1)


def make_run_id(seed: int) -> str:
    """
    Derive a short, deterministic run identifier from *seed*.

    The run ID is used as a human-readable reference in saved artefacts and
    the predict command so that the exact split can always be traced back.
    """
    rng = random.Random(seed)
    suffix = ''.join(rng.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
    return f'run_{seed}_{suffix}'


# ---------------------------------------------------------------------------
# Core split logic
# ---------------------------------------------------------------------------


def stratified_split(
    metadata: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 7,
) -> Tuple[pd.Index, pd.Index]:
    """
    Perform a single stratified train/test split on *metadata*.

    Parameters
    ----------
    metadata : pd.DataFrame
        Must contain at least a ``class`` column (string labels) and an
        integer ``record_index`` column that identifies each sample uniquely.
    test_size : float
        Fraction of samples held out for testing (default 0.20 → 80/20).
    seed : int
        Controls both StratifiedShuffleSplit and numpy RNG for full
        reproducibility.

    Returns
    -------
    train_idx, test_idx : pd.Index, pd.Index
        Integer-positional indices into *metadata* (i.e. suitable for
        ``metadata.iloc[train_idx]``).

    Raises
    ------
    ValueError
        If any class has fewer than 2 samples (stratification is impossible).
    """
    if 'class' not in metadata.columns:
        raise ValueError("metadata must contain a 'class' column.")

    metadata = metadata.reset_index(drop=True)

    labels = metadata['class'].astype(str)

    if labels.isna().any():
        raise ValueError("NaNs detected in 'class' after preprocessing.")

    labels = labels.values
    class_counts = pd.Series(labels).value_counts()
    if (class_counts < 2).any():
        tiny = class_counts[class_counts < 2].index.tolist()
        raise ValueError(
            f'Cannot stratify: class(es) {tiny} have fewer than 2 samples.'
        )

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    train_pos, test_pos = next(sss.split(np.zeros(len(metadata)), labels))
    return pd.Index(train_pos), pd.Index(test_pos)


def split_metadata(
    metadata: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 7,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Split *metadata* into train/test subsets and return both alongside
    a serialisable split-definition dictionary.

    The split definition records enough information to reconstruct the exact
    same partition from seed alone (via ``reconstruct_split``).

    Parameters
    ----------
    metadata : pd.DataFrame
        Full metadata table (output of ``generate_metadata``).
    test_size : float
        Held-out fraction (default 0.20).
    seed : int
        Random seed controlling the split.

    Returns
    -------
    train_meta, test_meta : pd.DataFrame
        Subsets of *metadata* for training and testing respectively.
    split_def : dict
        ``{'seed': int, 'test_size': float, 'run_id': str,
           'train_sample_ids': list[str], 'test_sample_ids': list[str]}``
    """
    train_idx, test_idx = stratified_split(
        metadata, test_size=test_size, seed=seed
    )
    train_meta = metadata.iloc[train_idx].reset_index(drop=True)
    test_meta = metadata.iloc[test_idx].reset_index(drop=True)
    split_def = {
        'seed': seed,
        'test_size': test_size,
        'run_id': make_run_id(seed),
        'train_sample_ids': train_meta['sample-id'].tolist(),
        'test_sample_ids': test_meta['sample-id'].tolist(),
    }

    return train_meta, test_meta, split_def


def reconstruct_split(
    metadata: pd.DataFrame,
    split_def: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconstruct the exact train/test split recorded in *split_def*.

    Reconstruction is done by matching ``sample-id`` values stored in
    *split_def*, which guarantees the same partition regardless of the
    order rows appear in *metadata*.

    Parameters
    ----------
    metadata : pd.DataFrame
        Full metadata table (must contain the same samples as at training
        time; ordering may differ).
    split_def : dict
        Artefact produced by ``split_metadata`` and persisted in the model
        file or a separate JSON.

    Returns
    -------
    train_meta, test_meta : pd.DataFrame

    Raises
    ------
    KeyError
        If ``sample-id`` values recorded in *split_def* are not all present
        in *metadata* (indicates a different dataset was supplied).
    """
    required_keys = {'train_sample_ids', 'test_sample_ids', 'seed', 'run_id'}
    missing = required_keys - set(split_def.keys())
    if missing:
        raise KeyError(
            f'split_def is missing required keys: {missing}. '
            'The model file may have been created with an older version of BITSER.'
        )

    all_ids = set(metadata['sample-id'])
    train_ids = split_def['train_sample_ids']
    test_ids = split_def['test_sample_ids']

    missing_train = [sid for sid in train_ids if sid not in all_ids]
    missing_test = [sid for sid in test_ids if sid not in all_ids]
    if missing_train or missing_test:
        raise KeyError(
            f'{len(missing_train)} train sample-id(s) and '
            f'{len(missing_test)} test sample-id(s) from the stored run '
            f'"{split_def["run_id"]}" were not found in the supplied dataset. '
            'Ensure you are using the same dataset that was used for training.'
        )

    train_meta = metadata[metadata['sample-id'].isin(train_ids)].reset_index(
        drop=True
    )
    test_meta = metadata[metadata['sample-id'].isin(test_ids)].reset_index(
        drop=True
    )

    return train_meta, test_meta

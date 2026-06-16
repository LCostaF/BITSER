"""
tests/test_split_manager.py
----------------------------
Tests for the new split_manager module.

Covers:
- Seed generation and run-ID derivation.
- Stratified 80/20 split correctness and reproducibility.
- Exact reconstruction of splits from stored split_def.
- Edge-case error handling (missing columns, too-small classes).
"""
import numpy as np
import pandas as pd
import pytest

from bitser.split_manager import (
    generate_seed,
    make_run_id,
    reconstruct_split,
    split_metadata,
    stratified_split,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(n_per_class: int = 50, n_classes: int = 2) -> pd.DataFrame:
    """Return a minimal metadata DataFrame with *n_classes* balanced classes."""
    rows = []
    for cls_idx in range(n_classes):
        cls_name = f'class{cls_idx}'
        for i in range(n_per_class):
            global_i = cls_idx * n_per_class + i
            rows.append(
                {
                    'sample-id': f'sample_{global_i}',
                    'fasta_path': f'sequences/file_{cls_idx}.fa',
                    'class': cls_name,
                    'record_index': i,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# generate_seed
# ---------------------------------------------------------------------------


def test_generate_seed_returns_int():
    s = generate_seed()
    assert isinstance(s, int)


def test_generate_seed_in_valid_range():
    for _ in range(20):
        s = generate_seed()
        assert 0 <= s <= 2**31 - 1


def test_generate_seed_randomness():
    seeds = {generate_seed() for _ in range(50)}
    # Very unlikely to get fewer than 10 unique values in 50 draws
    assert len(seeds) > 10


# ---------------------------------------------------------------------------
# make_run_id
# ---------------------------------------------------------------------------


def test_make_run_id_format():
    rid = make_run_id(42)
    assert rid.startswith('run_42_')
    assert len(rid) > len('run_42_')


def test_make_run_id_deterministic():
    assert make_run_id(99) == make_run_id(99)


def test_make_run_id_different_for_different_seeds():
    assert make_run_id(1) != make_run_id(2)


# ---------------------------------------------------------------------------
# stratified_split
# ---------------------------------------------------------------------------


def test_stratified_split_sizes():
    meta = _make_metadata(n_per_class=100)
    train_idx, test_idx = stratified_split(meta, test_size=0.2, seed=7)
    total = len(meta)
    assert len(train_idx) + len(test_idx) == total
    # Allow ±2 sample tolerance for rounding
    assert abs(len(test_idx) - 40) <= 2


def test_stratified_split_no_overlap():
    meta = _make_metadata(n_per_class=50)
    train_idx, test_idx = stratified_split(meta, test_size=0.2, seed=7)
    assert (
        len(set(train_idx) & set(test_idx)) == 0
    ), 'Train and test indices must not overlap'


def test_stratified_split_reproducible():
    meta = _make_metadata(n_per_class=60)
    t1, e1 = stratified_split(meta, test_size=0.2, seed=123)
    t2, e2 = stratified_split(meta, test_size=0.2, seed=123)
    assert list(t1) == list(t2)
    assert list(e1) == list(e2)


def test_stratified_split_different_seeds_differ():
    meta = _make_metadata(n_per_class=60)
    _, e1 = stratified_split(meta, test_size=0.2, seed=1)
    _, e2 = stratified_split(meta, test_size=0.2, seed=999)
    # Different seeds must (almost certainly) produce different splits
    assert set(e1) != set(e2)


def test_stratified_split_class_balance():
    """Each class should be represented in both train and test."""
    meta = _make_metadata(n_per_class=50, n_classes=3)
    train_idx, test_idx = stratified_split(meta, test_size=0.2, seed=7)
    train_classes = set(meta.iloc[train_idx]['class'].unique())
    test_classes = set(meta.iloc[test_idx]['class'].unique())
    expected = {'class0', 'class1', 'class2'}
    assert train_classes == expected
    assert test_classes == expected


def test_stratified_split_missing_class_column_raises():
    meta = pd.DataFrame({'sample-id': ['a', 'b'], 'record_index': [0, 1]})
    with pytest.raises(ValueError, match="'class' column"):
        stratified_split(meta)


def test_stratified_split_single_sample_class_raises():
    """A class with only one sample cannot be stratified."""
    rows = [
        {
            'sample-id': 'a',
            'fasta_path': 'f.fa',
            'class': 'A',
            'record_index': 0,
        },
        {
            'sample-id': 'b',
            'fasta_path': 'f.fa',
            'class': 'B',
            'record_index': 1,
        },
        {
            'sample-id': 'c',
            'fasta_path': 'f.fa',
            'class': 'B',
            'record_index': 2,
        },
    ]
    meta = pd.DataFrame(rows)
    with pytest.raises(ValueError, match='fewer than 2 samples'):
        stratified_split(meta)


# ---------------------------------------------------------------------------
# split_metadata
# ---------------------------------------------------------------------------


def test_split_metadata_returns_correct_types():
    meta = _make_metadata(n_per_class=50)
    train_m, test_m, split_def = split_metadata(meta, test_size=0.2, seed=7)
    assert isinstance(train_m, pd.DataFrame)
    assert isinstance(test_m, pd.DataFrame)
    assert isinstance(split_def, dict)


def test_split_metadata_split_def_keys():
    meta = _make_metadata(n_per_class=50)
    _, _, split_def = split_metadata(meta, seed=42)
    for key in (
        'seed',
        'test_size',
        'run_id',
        'train_sample_ids',
        'test_sample_ids',
    ):
        assert key in split_def, f'split_def missing key: {key}'


def test_split_metadata_seed_stored():
    meta = _make_metadata(n_per_class=50)
    _, _, split_def = split_metadata(meta, seed=1234)
    assert split_def['seed'] == 1234


def test_split_metadata_no_leakage():
    """train_sample_ids and test_sample_ids must be disjoint."""
    meta = _make_metadata(n_per_class=50)
    _, _, split_def = split_metadata(meta, seed=7)
    train_set = set(split_def['train_sample_ids'])
    test_set = set(split_def['test_sample_ids'])
    assert train_set.isdisjoint(
        test_set
    ), 'Train/test sample IDs must not overlap'


def test_split_metadata_covers_all_samples():
    meta = _make_metadata(n_per_class=50)
    _, _, split_def = split_metadata(meta, seed=7)
    total = len(split_def['train_sample_ids']) + len(
        split_def['test_sample_ids']
    )
    assert total == len(meta)


# ---------------------------------------------------------------------------
# reconstruct_split
# ---------------------------------------------------------------------------


def test_reconstruct_split_exact_match():
    meta = _make_metadata(n_per_class=50)
    train_orig, test_orig, split_def = split_metadata(meta, seed=7)

    # Shuffle metadata rows to simulate a different loading order
    shuffled = meta.sample(frac=1, random_state=99).reset_index(drop=True)
    train_rec, test_rec = reconstruct_split(shuffled, split_def)

    assert set(train_rec['sample-id']) == set(train_orig['sample-id'])
    assert set(test_rec['sample-id']) == set(test_orig['sample-id'])


def test_reconstruct_split_test_ids_preserved():
    meta = _make_metadata(n_per_class=60)
    _, _, split_def = split_metadata(meta, seed=42)
    _, test_rec = reconstruct_split(meta, split_def)
    assert set(test_rec['sample-id']) == set(split_def['test_sample_ids'])


def test_reconstruct_split_missing_keys_raises():
    meta = _make_metadata(n_per_class=20)
    bad_def = {'seed': 1}  # missing required keys
    with pytest.raises(KeyError, match='split_def is missing required keys'):
        reconstruct_split(meta, bad_def)


def test_reconstruct_split_wrong_dataset_raises():
    meta = _make_metadata(n_per_class=30)
    _, _, split_def = split_metadata(meta, seed=7)

    # Create a completely different dataset
    other_meta = _make_metadata(n_per_class=30)
    other_meta['sample-id'] = [f'OTHER_{i}' for i in range(len(other_meta))]

    with pytest.raises(KeyError, match='not found in the supplied dataset'):
        reconstruct_split(other_meta, split_def)


def test_reconstruct_split_no_leakage():
    """Reconstructed train and test sets must remain disjoint."""
    meta = _make_metadata(n_per_class=80)
    _, _, split_def = split_metadata(meta, seed=55)
    train_rec, test_rec = reconstruct_split(meta, split_def)
    train_ids = set(train_rec['sample-id'])
    test_ids = set(test_rec['sample-id'])
    assert train_ids.isdisjoint(test_ids)

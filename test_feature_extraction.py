# tests/test_feature_extraction.py
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from bitser.feature_extraction import extract_features_from_metadata


@pytest.fixture
def small_dataset(tmp_path: Path):
    """Create minimal dataset: one FASTA file + metadata with train/test split"""
    seq_dir = tmp_path / 'sequences'
    seq_dir.mkdir()

    metadata_rows = []

    # ── Create one FASTA file with 4 sequences ───────────────────────────────
    fasta_path = seq_dir / 'sample1.fasta'
    records = []

    for i in range(4):
        header = f'seq_{i} |class{i%2 + 1}| extra_info'
        seq_str = 'ACGT' * (10 + i * 5)  # different lengths
        records.append(
            SeqRecord(Seq(seq_str), id=f'seq_{i}', description=header)
        )
        metadata_rows.append(
            {
                'sample-id': f'seq_{i}',
                'fasta_path': f'sequences/{fasta_path.name}',
                'class': f'class{i%2 + 1}',
                'split': 'train' if i < 2 else 'test',
            }
        )

    from Bio import SeqIO

    SeqIO.write(records, fasta_path, 'fasta')

    # ── Write metadata ───────────────────────────────────────────────────────
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = tmp_path / 'metadata.tsv'
    metadata_df.to_csv(metadata_path, sep='\t', index=False)

    return {
        'dataset_dir': tmp_path,
        'metadata_path': metadata_path,
        'fasta_path': fasta_path,
        'expected_train_ids': {'seq_0', 'seq_1'},
        'expected_test_ids': {'seq_2', 'seq_3'},
        'expected_classes': ['class1', 'class2'],
    }


def test_extract_features_from_metadata_respects_split(small_dataset):
    data = small_dataset

    # ── Extract only train split ─────────────────────────────────────────────
    features_train, headers_train, seqs_train = extract_features_from_metadata(
        data['metadata_path'],
        split='train',
        flank=4,
        translate_sequences=False,
        n_jobs=1,
    )

    assert len(headers_train) == 2
    assert (
        set(h.split(' ')[0] for h in headers_train)
        == data['expected_train_ids']
    )
    assert features_train.shape[0] == 2
    assert features_train.shape[1] == 256 + 2 + 1  # Hist + BWS + BWP + CLASS

    # ── Extract only test split ──────────────────────────────────────────────
    features_test, headers_test, seqs_test = extract_features_from_metadata(
        data['metadata_path'],
        split='test',
        flank=4,
        translate_sequences=False,
        n_jobs=1,
    )

    assert len(headers_test) == 2
    assert (
        set(h.split(' ')[0] for h in headers_test) == data['expected_test_ids']
    )


def test_extract_features_all_when_no_split_filter(small_dataset):
    data = small_dataset

    features, headers, seqs = extract_features_from_metadata(
        data['metadata_path'],
        split=None,  # ← no filter
        flank=4,
        translate_sequences=False,
        n_jobs=1,
    )

    assert len(headers) == 4
    assert features.shape[0] == 4

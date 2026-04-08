# tests/test_feature_extraction.py
from pathlib import Path

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
    fasta_file = seq_dir / 'sample1.fasta'
    records = []

    for i in range(4):
        header = f'seq_{i} |class{i%2 + 1}| extra_info'
        seq_str = 'ACGT' * (10 + i * 5)
        records.append(
            SeqRecord(Seq(seq_str), id=f'seq_{i}', description=header)
        )
        metadata_rows.append(
            {
                'sample-id': f'seq_{i}',
                'fasta_path': f'sequences/{fasta_file.name}',  # relative to tmp_path
                'class': f'class{i%2 + 1}',
                'split': 'train' if i < 2 else 'test',
                'record_index': i,
            }
        )

    from Bio import SeqIO

    SeqIO.write(records, fasta_file, 'fasta')

    # ── Write metadata to tmp_path/metadata.tsv ──────────────────────────────
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = tmp_path / 'metadata.tsv'
    metadata_df.to_csv(metadata_path, sep='\t', index=False)

    return {
        'dataset_dir': tmp_path,  # kept for possible future use
        'metadata_path': metadata_path,  # ← This must be the file, not a directory
        'fasta_path': fasta_file,
        'expected_train_ids': {'seq_0', 'seq_1'},
        'expected_test_ids': {'seq_2', 'seq_3'},
        'expected_classes': ['class1', 'class2'],
    }


def test_extract_features_from_metadata_respects_split(small_dataset):
    data = small_dataset

    # Train split
    features_train, headers_train, seqs_train = extract_features_from_metadata(
        data['dataset_dir'],  # ← Must be the .tsv file (not the folder)
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
    assert (
        features_train.shape[1] == 256 + 2 + 1
    )   # 256 (hist) + bws + bwp + class

    # Test split
    features_test, headers_test, seqs_test = extract_features_from_metadata(
        data['dataset_dir'],  # ← Same here
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
        data['dataset_dir'],  # ← Correct argument
        split=None,
        flank=4,
        translate_sequences=False,
        n_jobs=1,
    )

    assert len(headers) == 4
    assert features.shape[0] == 4

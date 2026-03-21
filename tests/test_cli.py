# tests/test_cli.py
import re
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from typer.testing import CliRunner

from bitser import __version__
from bitser.cli import app

runner = CliRunner()


@pytest.fixture
def minimal_dataset(tmp_path: Path) -> Generator[dict, None, None]:
    """
    Creates:
    - dataset/
      - sequences/
        - seqs1.fasta   (4 sequences: 2×classA train, 2×classB test)
    """
    dataset_dir = tmp_path / 'dataset'
    seq_dir = dataset_dir / 'sequences'
    seq_dir.mkdir(parents=True)

    fasta_content = []
    for i in range(4):
        cls = f'class{i % 2 + 1}'
        header = f'seq{i} organism=TestOrganism |{cls}'
        seq = Seq('ACGT' * (20 + i * 5))
        rec = SeqRecord(seq, id=f'seq{i}', description=header)
        fasta_content.append(rec)

    fasta_path = seq_dir / 'seqs1.fasta'
    SeqIO.write(fasta_content, fasta_path, 'fasta')

    yield {
        'dataset_dir': dataset_dir,
        'sequences_dir': seq_dir,
        'fasta_path': fasta_path,
        'metadata_expected_path': dataset_dir / 'metadata.tsv',
    }


def test_version_flag():
    result = runner.invoke(app, ['--version'])
    assert result.exit_code == 0

    # Clean ANSI codes if present
    clean = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', result.stdout)
    assert __version__ in clean
    assert 'BITSER version:' in clean


def test_help_output():
    result = runner.invoke(app, ['--help'])
    assert result.exit_code == 0
    assert (
        'BITSER - Bioinformatics Tool for Sequence Classification'
        in result.stdout
    )
    assert 'metadata' in result.stdout
    assert 'train' in result.stdout
    assert 'predict' in result.stdout


def test_metadata_command_success(minimal_dataset):
    ds = minimal_dataset

    result = runner.invoke(
        app,
        [
            'metadata',
            '--dataset',
            str(ds['dataset_dir']),
            '--class-delim',
            '|',
            '--class-which',
            '-1',
            '--train-count',
            '1',  # 1 train per class
            '--seed',
            '42',
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert 'metadata.tsv created' in result.stdout
    assert '✓' in result.stdout

    metadata_path = ds['metadata_expected_path']
    assert metadata_path.is_file()

    df = pd.read_csv(metadata_path, sep='\t')
    assert list(df.columns) == ['sample-id', 'fasta_path', 'class', 'split']
    assert len(df) == 4
    assert set(df['class'].unique()) == {'class1', 'class2'}
    assert df['split'].value_counts().to_dict() == {
        'train': 2,
        'test': 2,
    }  # 1 per class

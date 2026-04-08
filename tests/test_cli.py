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
    """Creates a minimal dataset for testing."""
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
        'metadata_path': dataset_dir / 'metadata.tsv',
    }


def test_version_flag():
    result = runner.invoke(app, ['--version'])
    assert result.exit_code == 0

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
            '1',
            '--seed',
            '42',
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert 'metadata.tsv created' in result.stdout


# ==================== ERROR CASE TESTS ====================


def test_metadata_missing_required_args():
    """Typer shows its own error when a required Option is missing (exit code 2)."""
    result = runner.invoke(app, ['metadata'])

    assert result.exit_code == 2
    assert "Missing option '--dataset' / '-d'" in result.stdout
    assert 'Usage: root metadata [OPTIONS]' in result.stdout


def test_metadata_invalid_train_count(minimal_dataset):
    """Test custom validation that runs *after* Typer parsing."""
    ds = minimal_dataset
    result = runner.invoke(
        app,
        [
            'metadata',
            '--dataset',
            str(ds['dataset_dir']),
            '--class-delim',
            '|',
            '--train-count',
            '0',  # triggers custom validation
        ],
    )
    assert result.exit_code == 1
    assert '--train-count must be a positive integer' in result.stdout


def test_train_missing_required_args():
    result = runner.invoke(app, ['train'])
    assert result.exit_code == 2
    assert "Missing option '--input' / '-i'" in result.stdout


def test_predict_missing_required_args():
    result = runner.invoke(app, ['predict'])
    assert result.exit_code == 2
    assert "Missing option '--model' / '-m'" in result.stdout

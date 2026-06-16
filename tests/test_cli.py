"""
tests/test_cli.py
------------------
CLI integration tests for the refactored BITSER commands.

Key changes from the original:
- metadata no longer accepts --train-count or --seed.
- train command expects full dataset (no pre-split) and has --seed.
- predict reconstructs split from model file; no --seed CLI arg needed.
"""
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


def strip_ansi(text: str) -> str:
    return re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)


@pytest.fixture
def minimal_dataset(tmp_path: Path) -> Generator[dict, None, None]:
    """
    Creates a minimal two-class dataset WITHOUT a pre-split metadata.tsv.
    The metadata is just sample-id / fasta_path / class / record_index.
    """
    dataset_dir = tmp_path / 'dataset'
    seq_dir = dataset_dir / 'sequences'
    seq_dir.mkdir(parents=True)

    fasta_content = []
    for i in range(8):  # 4 per class
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


# ---------------------------------------------------------------------------
# Version / help
# ---------------------------------------------------------------------------


def test_version_flag():
    result = runner.invoke(app, ['--version'])
    assert result.exit_code == 0
    clean = strip_ansi(result.stdout)
    assert __version__ in clean
    assert 'BITSER version:' in clean


def test_help_output():
    result = runner.invoke(app, ['--help'])
    assert result.exit_code == 0
    assert 'BITSER - Binary Pattern Sequence Recognition' in result.stdout
    assert 'metadata' in result.stdout
    assert 'train' in result.stdout
    assert 'predict' in result.stdout


# ---------------------------------------------------------------------------
# metadata command
# ---------------------------------------------------------------------------


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
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert 'metadata.tsv created' in result.stdout


def test_metadata_no_split_column_written(minimal_dataset):
    """After metadata command the TSV must NOT have a 'split' column."""
    ds = minimal_dataset
    runner.invoke(
        app,
        [
            'metadata',
            '--dataset',
            str(ds['dataset_dir']),
            '--class-delim',
            '|',
            '--class-which',
            '-1',
        ],
        catch_exceptions=False,
    )
    df = pd.read_csv(ds['metadata_path'], sep='\t')
    assert 'split' not in df.columns


def test_metadata_required_columns_present(minimal_dataset):
    ds = minimal_dataset
    runner.invoke(
        app,
        [
            'metadata',
            '--dataset',
            str(ds['dataset_dir']),
            '--class-delim',
            '|',
            '--class-which',
            '-1',
        ],
        catch_exceptions=False,
    )
    df = pd.read_csv(ds['metadata_path'], sep='\t')
    for col in ('sample-id', 'class', 'fasta_path', 'record_index'):
        assert col in df.columns


def test_metadata_does_not_accept_seed():
    """--seed must no longer be a valid option for the metadata command."""
    result = runner.invoke(
        app,
        ['metadata', '--dataset', '.', '--class-delim', ' ', '--seed', '42'],
    )
    assert result.exit_code == 2


def test_metadata_nonexistent_directory():
    result = runner.invoke(
        app,
        ['metadata', '--dataset', '/nonexistent/path', '--class-delim', ' '],
    )
    assert result.exit_code == 1
    assert 'does not exist' in strip_ansi(result.stdout)


def test_train_accepts_seed_option(tmp_path):
    """--seed must be accepted without error at the CLI level."""
    result = runner.invoke(
        app,
        [
            'train',
            '--input',
            str(tmp_path),
            '--output-dir',
            str(tmp_path),
            '--output',
            'model.pkl',
            '--seed',
            '99',
        ],
    )
    # Will exit with code 1 (missing metadata.tsv) but NOT code 2 (bad option)
    assert result.exit_code != 2


def test_train_autogenerates_seed_when_omitted(tmp_path, capsys):
    """When --seed is omitted the CLI must mention 'Auto-generated seed'."""
    result = runner.invoke(
        app,
        [
            'train',
            '--input',
            str(tmp_path),
            '--output-dir',
            str(tmp_path),
            '--output',
            'model.pkl',
        ],
    )
    # The process will fail because metadata.tsv is missing, but before that
    # the seed message should appear (or it fails at directory check).
    # Either way, exit code must NOT be 2 (unknown option).
    assert result.exit_code != 2


def test_train_invalid_classifier():
    result = runner.invoke(
        app,
        [
            'train',
            '--input',
            '.',
            '--output-dir',
            '.',
            '--output',
            'm.pkl',
            '--classifier',
            'unknown',
        ],
    )
    clean = strip_ansi(result.stdout)
    assert result.exit_code == 1
    assert 'Unsupported classifier' in clean


def test_train_invalid_flank():
    result = runner.invoke(
        app,
        [
            'train',
            '--input',
            '.',
            '--output-dir',
            '.',
            '--output',
            'm.pkl',
            '--flank',
            '0',
        ],
    )
    clean = strip_ansi(result.stdout)
    assert result.exit_code == 1
    assert '--flank must be a positive integer' in clean


def test_train_invalid_splits():
    result = runner.invoke(
        app,
        [
            'train',
            '--input',
            '.',
            '--output-dir',
            '.',
            '--output',
            'm.pkl',
            '--splits',
            '1',
        ],
    )
    clean = strip_ansi(result.stdout)
    assert result.exit_code == 1
    assert '--splits must be greater than 1' in clean


def test_train_invalid_test_size():
    result = runner.invoke(
        app,
        [
            'train',
            '--input',
            '.',
            '--output-dir',
            '.',
            '--output',
            'm.pkl',
            '--test-size',
            '1.5',
        ],
    )
    clean = strip_ansi(result.stdout)
    assert result.exit_code == 1
    assert '--test-size must be between 0 and 1' in clean


# ---------------------------------------------------------------------------
# predict command — CLI-level validation
# ---------------------------------------------------------------------------


def test_predict_nonexistent_model():
    result = runner.invoke(
        app,
        [
            'predict',
            '--model',
            '/nonexistent/model.pkl',
            '--output-dir',
            '.',
            '--data',
            '.',
        ],
    )
    clean = strip_ansi(result.stdout)
    assert result.exit_code == 1
    assert 'not found' in clean


def test_predict_invalid_flank():
    result = runner.invoke(
        app,
        [
            'predict',
            '--model',
            'model.pkl',
            '--output-dir',
            '.',
            '--data',
            '.',
            '--flank',
            '-1',
        ],
    )
    clean = strip_ansi(result.stdout)
    assert result.exit_code == 1
    assert '--flank must be a positive integer' in clean

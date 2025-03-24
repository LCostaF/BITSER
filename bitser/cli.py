import pytest

from rich.console import Console
from typer import Context, Exit, Option, Typer
from typing_extensions import Annotated

from bitser import __version__
from bitser.main import main as _main

app = Typer(rich_markup_mode='rich')
console = Console()


def show_version(flag):
    if flag:
        print(f'BITSER version: {__version__}')
        raise Exit(code=0)


@app.callback(invoke_without_command=True)
def main(
    ctx: Context,
    version: bool = Option(
        False,
        '--version',
        '-v',
        callback=show_version,
        is_eager=True,
        help='Show version and exit.',
    ),
):
    message = (
        'Welcome to BITSER! Type bitser --help to see the available commands.'
    )
    if ctx.invoked_subcommand:
        return
    console.print(message)


@app.command()
def run(
        train_dir: Annotated[
            str,
            Option(
                '--train-dir',
                help='Path to directory containing the training data in FASTA format.',
            ),
        ],
        test_dir: Annotated[
            str,
            Option(
                '--test-dir',
                help='Path to directory containing the test data in FASTA format. If None, cross-validation is performed.',
            ),
        ] = None,
        translate_sequences: Annotated[
            bool,
            Option(
                '--translate-sequences',
                help='Boolean to translate sequences or not.',
            ),
        ] = False,
        classifier_type: Annotated[
            str,
            Option(
                '--classifier-type',
                help='Flag for classifier type.',
            ),
        ] = 'xgb',
        flank: Annotated[
            int,
            Option(
                '--flank',
                help='Size of the sliding window that runs through the sequence.',
            ),
        ] = 8,
        n_splits: Annotated[
            int,
            Option(
                '--n-splits',
                help='Number of splits for k-fold cross-validation.',
            ),
        ] = 10,
        n_repeats: Annotated[
            int,
            Option(
                '--n-repeats',
                help='Number of times to repeat the cross-validation process.',
            ),
        ] = 10,
        seed: Annotated[
            int,
            Option(
                '--seed',
                help='Random seed for reproducibility.',
            ),
        ] = 7,
):
    """
    Runs BITSER feature extraction and classification on sequence data.

    This command performs feature extraction and trains a classifier on the provided data.
    If test_dir is provided, it evaluates the model on the test data.
    Otherwise, it performs cross-validation on the training data.
    """
    _main(
        train_dir=train_dir,
        test_dir=test_dir,
        translate_sequences=translate_sequences,
        classifier_type=classifier_type,
        flank=flank,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
    )


@app.command()
def test(
        verbose: Annotated[
            bool,
            Option(
                "--verbose", "-v",
                help="Run tests in verbose mode."
            ),
        ] = False,
        test_path: Annotated[
            str,
            Option(
                "--path", "-p",
                help="Path to the test directory or specific test file to run.",
            ),
        ] = "tests/",
):
    """
    Run the project tests using pytest.

    This command executes all tests in the specified directory (defaults to 'tests/').
    """

    args = [test_path]
    if verbose:
        args.append("-v")

    exit_code = pytest.main(args)
    if exit_code != 0:
        raise Exit(code=exit_code)

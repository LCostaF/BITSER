import time
from pathlib import Path

import pandas as pd
import pytest
from rich.console import Console
from rich.progress import track
from typer import Context, Exit, Option, Typer
from typing_extensions import Annotated

from bitser import __version__
from bitser.data_preprocessing import prepare_dataframe
from bitser.feature_extraction import extract_features_from_metadata
from bitser.file_utils import save_output_to_file, save_prediction_report
from bitser.metadata_utils import generate_metadata
from bitser.model_training import (
    load_model,
    predict_and_evaluate,
    save_model,
    train_classification_model,
)

app = Typer(
    rich_markup_mode='rich',
    help="""BITSER - Bioinformatics Tool for Sequence Classification

Examples:
  [bold]Train a model:[/bold]
  bitser train --input training_data/ --output model.pkl

  [bold]Test sequences:[/bold]
  bitser predict --model model.pkl --data test_sequences/

  [bold]Quick start:[/bold]
  bitser train -i training/ -o results/model.pkl -f 8
""",
)
console = Console()


def show_version(flag):
    if flag:
        console.print(f'[bold]BITSER version:[/bold] {__version__}')
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
    if ctx.invoked_subcommand:
        return
    console.print(
        '[bold]Welcome to BITSER![/bold] Type [cyan]bitser --help[/cyan] to see available commands.'
    )


@app.command()
def metadata(
    dataset: Annotated[
        str,
        Option(
            '--dataset',
            '-d',
            help='Dataset directory containing sequences/ folder',
        ),
    ],
    class_delim: Annotated[
        str,
        Option(
            '--class-delim',
            '-delim',
            help='Delimiter string before the class label (required). Examples: " ", "|", "genotype "',
        ),
    ],
    train_count: Annotated[
        int,
        Option(
            '--train-count',
            '-n',
            help='Number of sequences per class to use for training',
        ),
    ],
    class_which: Annotated[
        int,
        Option(
            '--class-which',
            '-which',
            help='Which occurrence of the delimiter to use (1 = first, -1 = last, default: 1)',
        ),
    ] = 1,
    seed: Annotated[
        int,
        Option(
            '--seed',
            help='Random seed for reproducibility',
        ),
    ] = 7,
):
    """
    Generate metadata.tsv describing dataset splits using FASTA header parsing.

    You must specify --class-delim to tell the tool how to find the class label.
    The class token is automatically cleaned to contain only alphanumeric characters.
    """
    # === Early validation for missing required arguments ===
    if not dataset:
        console.print(
            '[red bold]Error:[/red bold] --dataset (-d) is required.'
        )
        raise Exit(code=1)
    if not class_delim:
        console.print('[red bold]Error:[/red bold] --class-delim is required.')
        console.print('Examples:')
        console.print('  --class-delim " " --class-which 1')
        console.print('  --class-delim "|" --class-which -1')
        console.print('  --class-delim "genotype " --class-which 1')
        raise Exit(code=1)

    # === Early validation for invalid argument values ===
    if train_count <= 0:
        console.print(
            '[red bold]Error:[/red bold] --train-count must be a positive integer (> 0).'
        )
        raise Exit(code=1)
    if class_which == 0:
        console.print(
            '[red bold]Error:[/red bold] --class-which cannot be 0 (use 1 for first occurrence or -1 for last).'
        )
        raise Exit(code=1)

    # === Early validation for dataset path issues ===
    dataset_path = Path(dataset).resolve()
    if not dataset_path.exists():
        console.print(
            f'[red bold]Error:[/red bold] Dataset directory does not exist: [yellow]{dataset}[/yellow]'
        )
        raise Exit(code=1)
    if not dataset_path.is_dir():
        console.print(
            f'[red bold]Error:[/red bold] Path is not a directory: [yellow]{dataset}[/yellow]'
        )
        raise Exit(code=1)

    console.print('[cyan]Generating metadata table...[/cyan]')
    console.print(f'  delimiter   : [yellow]{class_delim!r}[/yellow]')
    console.print(f'  occurrence  : [yellow]{class_which}[/yellow]')

    try:
        path = generate_metadata(
            dataset,
            train_count=train_count,
            seed=seed,
            class_delim=class_delim,
            class_which=class_which,
        )
    except FileNotFoundError as e:
        console.print(
            '[red bold]Error:[/red bold] sequences/ folder not found in dataset directory.'
        )
        raise Exit(code=1)
    except ValueError as e:
        err_msg = str(e).lower()
        if 'no valid sequences' in err_msg or 'header parsing' in err_msg:
            console.print(
                '[red bold]Error:[/red bold] No valid sequences found after header parsing (check --class-delim, --class-which, or FASTA headers).'
            )
        elif 'fewer than 2 valid classes' in err_msg:
            console.print(
                '[red bold]Error:[/red bold] Dataset has fewer than 2 valid classes. Classification requires at least 2 classes.'
            )
        else:
            console.print(f'[red bold]Error:[/red bold] {str(e)}')
        raise Exit(code=1)
    except OSError as e:
        console.print(
            '[red bold]Error:[/red bold] Failure writing metadata.tsv (permissions or disk issues).'
        )
        raise Exit(code=1)
    except Exception as e:
        err_str = str(e).lower()
        if (
            'fasta' in err_str
            or 'parse' in err_str
            or 'seqio' in err_str
            or 'unexpected' in err_str
        ):
            console.print(
                '[red bold]Error:[/red bold] Corrupted, unreadable FASTA files or invalid FASTA format in sequences/.'
            )
        elif 'header' in err_str or 'delim' in err_str or 'class' in err_str:
            console.print(
                '[red bold]Error:[/red bold] Header parsing issues (delimiter not found in headers, invalid class-which, or extracted class empty/invalid after cleaning).'
            )
        else:
            console.print(
                f'[red bold]Error:[/red bold] Unexpected error during metadata generation: {str(e)}'
            )
        raise Exit(code=1)

    console.print(f'[bold green]✓ metadata.tsv created at {path}[/bold green]')


@app.command()
def train(
    input: Annotated[
        str,
        Option(
            '--input',
            '-i',
            help='Dataset directory (must contain metadata.tsv and sequences/ folder)',
        ),
    ],
    output_dir: Annotated[
        str,
        Option(
            '--output-dir',
            '-dir',
            help='Directory where all outputs will be saved',
        ),
    ],
    output: Annotated[
        str,
        Option(
            '--output',
            '-o',
            help='Path to save the trained model (e.g., "model.pkl")',
        ),
    ] = 'model.pkl',
    classifier: Annotated[
        str,
        Option(
            '--classifier',
            '-c',
            help='Classifier algorithm: "rf" (Random Forest) or "xgb" (XGBoost)',
        ),
    ] = 'xgb',
    flank: Annotated[
        int,
        Option(
            '--flank',
            '-f',
            help='Sliding window size for feature extraction (default: 8)',
        ),
    ] = 8,
    translate: Annotated[
        bool,
        Option(
            '--translate/--no-translate',
            help='Translate nucleotide sequences to proteins',
        ),
    ] = False,
    splits: Annotated[
        int,
        Option(
            '--splits',
            '-s',
            help='Number of cross-validation folds (default: 10)',
        ),
    ] = 10,
    repeats: Annotated[
        int,
        Option(
            '--repeats',
            '-r',
            help='Cross-validation repetitions (default: 10)',
        ),
    ] = 10,
    seed: Annotated[
        int,
        Option(
            '--seed',
            help='Random seed for reproducibility (default: 7)',
        ),
    ] = 7,
):
    """
    Train a classification model from sequence data.

    The training process includes:
    1. Feature extraction using sliding windows
    2. Model training with cross-validation
    3. Saving the trained model for future use
    """
    # === Early validation for missing required arguments ===
    if not input:
        console.print('[red bold]Error:[/red bold] --input (-i) is required.')
        raise Exit(code=1)
    if not output_dir:
        console.print(
            '[red bold]Error:[/red bold] --output-dir (-dir) is required.'
        )
        raise Exit(code=1)

    # === Early validation for invalid argument values ===
    if classifier not in {'xgb', 'rf', 'svm', 'mlp', 'nb'}:
        console.print(
            f'[red bold]Error:[/red bold] Unsupported classifier "{classifier}". Must be one of: xgb, rf, svm, mlp, nb.'
        )
        raise Exit(code=1)
    if flank <= 0:
        console.print(
            '[red bold]Error:[/red bold] --flank must be a positive integer (> 0).'
        )
        raise Exit(code=1)
    if splits <= 1:
        console.print(
            '[red bold]Error:[/red bold] --splits must be greater than 1.'
        )
        raise Exit(code=1)
    if repeats <= 0:
        console.print(
            '[red bold]Error:[/red bold] --repeats must be a positive integer (> 0).'
        )
        raise Exit(code=1)

    # === Early validation for dataset path and required structure ===
    input_path = Path(input).resolve()
    if not input_path.exists():
        console.print(
            f'[red bold]Error:[/red bold] Dataset directory does not exist: [yellow]{input}[/yellow]'
        )
        raise Exit(code=1)
    if not input_path.is_dir():
        console.print(
            f'[red bold]Error:[/red bold] Path is not a directory: [yellow]{input}[/yellow]'
        )
        raise Exit(code=1)
    if not (input_path / 'metadata.tsv').is_file():
        console.print(
            '[red bold]Error:[/red bold] metadata.tsv is missing in the dataset directory.'
        )
        raise Exit(code=1)
    if not (input_path / 'sequences').is_dir():
        console.print(
            '[red bold]Error:[/red bold] sequences/ folder not found in dataset directory.'
        )
        raise Exit(code=1)

    start_time = time.time()
    console.print(
        f'[bold]Training model with {classifier} classifier...[/bold]'
    )

    try:
        # Extract features with progress indication
        console.print('[cyan]Extracting features...[/cyan]')
        train_features, _, _ = extract_features_from_metadata(
            input,
            split='train',
            flank=flank,
            translate_sequences=translate,
        )
        console.print(
            '[bold green]✓ Feature extraction complete![/bold green]'
        )

        console.print('[cyan]Preparing dataframe...[/cyan]')
        train_df, train_classes, name_class = prepare_dataframe(train_features)
        console.print('[bold green]✓ Dataframe prepared![/bold green]')

        console.print('[cyan]Training model...[/cyan]')
        (
            classifier_model,
            min_max_scaler,
            label_encoder,
            _,
            output_text,
        ) = train_classification_model(
            train_df,
            train_classes,
            classifier_type=classifier,
            n_splits=splits,
            n_repeats=repeats,
            seed=seed,
            perform_cv=True,
        )
        console.print('[bold green]✓ Finished training model![/bold green]')

        save_output_to_file(output_text, classifier, output_dir)

        save_model(
            classifier_model,
            min_max_scaler,
            label_encoder,
            None,
            output,
            name_class=name_class,
            output_text=output_text,
            train_df_columns=train_df.columns.tolist(),
        )

        console.print(
            f'[bold green]✓ Success![/bold green] Model saved to [cyan]{output}[/cyan]'
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time, 60)

        console.print(
            f'[bold]Total execution time:[/bold] {int(minutes)} minutes {seconds:.2f} seconds'
        )
    except FileNotFoundError as e:
        console.print(
            '[red bold]Error:[/red bold] Missing FASTA files referenced in metadata or unreadable FASTA files.'
        )
        raise Exit(code=1)
    except ValueError as e:
        msg = str(e).lower()
        if 'empty' in msg or 'dataframe' in msg:
            console.print(
                '[red bold]Error:[/red bold] Empty dataframe after feature extraction.'
            )
        elif 'nan' in msg or 'invalid' in msg:
            console.print(
                '[red bold]Error:[/red bold] Data contains NaN or invalid values.'
            )
        elif 'short' in msg or 'flank' in msg:
            console.print(
                '[red bold]Error:[/red bold] Sequence too short for chosen flank size.'
            )
        else:
            console.print(f'[red bold]Error:[/red bold] {str(e)}')
        raise Exit(code=1)
    except OSError as e:
        console.print(
            '[red bold]Error:[/red bold] Failure writing model file or training output logs (permissions or disk issues).'
        )
        raise Exit(code=1)
    except Exception as e:
        msg = str(e).lower()
        if 'xgboost' in msg or 'not installed' in msg or 'import' in msg:
            console.print(
                '[red bold]Error:[/red bold] Required library not installed (e.g., XGBoost for xgb classifier).'
            )
        elif 'cross-validation' in msg or 'splits' in msg or 'samples' in msg:
            console.print(
                '[red bold]Error:[/red bold] Cross-validation infeasible (too few samples per class for the number of splits/repeats).'
            )
        elif 'feature' in msg or 'columns' in msg or 'flank' in msg:
            console.print(
                '[red bold]Error:[/red bold] Feature extraction or data preparation failure (inconsistent feature vectors).'
            )
        else:
            console.print(
                f'[red bold]Error:[/red bold] Model training failure: {str(e)}'
            )
        raise Exit(code=1)


@app.command(name='predict')
def test(
    model: Annotated[
        str,
        Option(
            '--model',
            '-m',
            help='Path to trained model file (e.g., "model.pkl")',
        ),
    ],
    output_dir: Annotated[
        str,
        Option(
            '--output-dir',
            '-dir',
            help='Directory where prediction outputs will be saved',
        ),
    ],
    data: Annotated[
        str,
        Option(
            '--data',
            '-d',
            help='Dataset directory (must contain metadata.tsv and sequences/ folder)',
        ),
    ] = None,
    flank: Annotated[
        int,
        Option(
            '--flank',
            '-f',
            help='Sliding window size (must match training setting)',
        ),
    ] = 8,
    translate: Annotated[
        bool,
        Option(
            '--translate/--no-translate',
            help='Translate nucleotide sequences to proteins',
        ),
    ] = False,
):
    """
    Predict classes for new sequences using a trained model.

    Output includes:
    - Classification accuracy
    - Per-class performance metrics
    - Confusion matrix (if applicable)
    """
    test_headers = None
    test_sequences = None

    # === Early validation for missing required arguments ===
    if not model:
        console.print('[red bold]Error:[/red bold] --model (-m) is required.')
        raise Exit(code=1)

    if not output_dir:
        console.print('[red bold]Error:[/red bold] --output-dir is required.')
        raise Exit(code=1)

    # === Early validation for invalid argument values ===
    if flank <= 0:
        console.print(
            '[red bold]Error:[/red bold] --flank must be a positive integer (> 0).'
        )
        raise Exit(code=1)

    start_time = time.time()
    console.print(f'[bold]Loading model from {model}...[/bold]')

    try:
        model_data = load_model(model)
        console.print(
            f'[bold green]✓ {model} loaded successfully![/bold green]'
        )
    except FileNotFoundError:
        console.print(
            f'[red bold]Error:[/red bold] Model file not found: [yellow]{model}[/yellow]'
        )
        raise Exit(code=1)
    except Exception as e:
        console.print(
            '[red bold]Error:[/red bold] Model file is corrupted, incompatible, or missing expected keys.'
        )
        raise Exit(code=1)

    if data:
        # === Early validation for test dataset path and required structure ===
        data_path = Path(data).resolve()
        if not data_path.exists():
            console.print(
                f'[red bold]Error:[/red bold] Test dataset directory does not exist: [yellow]{data}[/yellow]'
            )
            raise Exit(code=1)
        if not data_path.is_dir():
            console.print(
                f'[red bold]Error:[/red bold] Path is not a directory: [yellow]{data}[/yellow]'
            )
            raise Exit(code=1)
        if not (data_path / 'metadata.tsv').is_file():
            console.print(
                '[red bold]Error:[/red bold] metadata.tsv is missing in the test dataset directory.'
            )
            raise Exit(code=1)
        if not (data_path / 'sequences').is_dir():
            console.print(
                '[red bold]Error:[/red bold] sequences/ folder not found in test dataset directory.'
            )
            raise Exit(code=1)

        console.print('[cyan]Processing test sequences...[/cyan]')
        try:
            (
                test_features,
                test_headers,
                test_sequences,
            ) = extract_features_from_metadata(
                data,
                split='test',
                flank=flank,
                translate_sequences=translate,
            )
            test_df, test_classes, _ = prepare_dataframe(test_features)
            console.print(
                '[bold green]✓ Test sequences processed![/bold green]'
            )
        except FileNotFoundError as e:
            console.print(
                '[red bold]Error:[/red bold] Missing FASTA files referenced in metadata or unreadable FASTA files.'
            )
            raise Exit(code=1)
        except ValueError as e:
            msg = str(e).lower()
            if 'empty' in msg or 'dataframe' in msg:
                console.print(
                    '[red bold]Error:[/red bold] Empty dataframe after feature extraction.'
                )
            elif 'nan' in msg or 'invalid' in msg:
                console.print(
                    '[red bold]Error:[/red bold] Data contains NaN or invalid values.'
                )
            elif 'short' in msg or 'flank' in msg:
                console.print(
                    '[red bold]Error:[/red bold] Sequence too short for chosen flank size.'
                )
            elif 'translation' in msg or 'nucleotide' in msg:
                console.print(
                    '[red bold]Error:[/red bold] Translation error (invalid nucleotide sequences).'
                )
            elif 'record' in msg or 'index' in msg or 'mismatch' in msg:
                console.print(
                    '[red bold]Error:[/red bold] Record index mismatch in metadata.'
                )
            else:
                console.print(f'[red bold]Error:[/red bold] {str(e)}')
            raise Exit(code=1)
        except Exception as e:
            console.print(
                f'[red bold]Error:[/red bold] Feature extraction failure: {str(e)}'
            )
            raise Exit(code=1)
    else:
        if 'test_data' not in model_data:
            console.print(
                '[red]Error:[/red] No test data provided and no saved test data found in model!'
            )
            raise Exit(code=1)
        test_df, test_classes = model_data['test_data']

    classifier_type = type(model_data['classifier']).__name__.lower()

    console.print('[cyan]Running predictions...[/cyan]')
    try:
        _, _, _, complete_output, predictions, y_test = predict_and_evaluate(
            model_data['classifier'],
            model_data['scaler'],
            model_data['encoder'],
            test_df,
            test_classes,
            model_data.get('name_class', []),
            train_df=None,
            previous_output=model_data.get('output_text', ''),
            classifier_type=classifier_type,
            validation_df=None,
            validation_classes=None,
            save_files=True,
        )
        console.print(
            f'[bold green]✓ Prediction complete![/bold green] Results saved to output files'
        )
    except Exception as e:
        msg = str(e).lower()
        if (
            'shape' in msg
            or 'mismatch' in msg
            or 'columns' in msg
            or 'flank' in msg
            or 'feature' in msg
        ):
            console.print(
                '[red bold]Error:[/red bold] Feature mismatch with trained model (different flank or feature columns).'
            )
        else:
            console.print(
                f'[red bold]Error:[/red bold] Prediction execution failure: {str(e)}'
            )
        raise Exit(code=1)

    # Generate and save CSV report
    console.print('[cyan]Generating prediction report...[/cyan]')
    try:
        report_df = pd.DataFrame(
            {
                'True Class': test_classes,
                'Label': test_headers
                if test_headers is not None
                else [None] * len(test_classes),
                'Sequence': test_sequences
                if test_sequences is not None
                else [None] * len(test_classes),
                'Predicted Class': predictions,
            }
        )

        csv_path = save_prediction_report(
            report_df, classifier_type, output_dir
        )
        console.print(
            f'[bold green]✓ Prediction report saved to {csv_path}![/bold green]'
        )
    except OSError as e:
        console.print(
            '[red bold]Error:[/red bold] Failure writing prediction report (permissions or disk issues).'
        )
        raise Exit(code=1)
    except Exception as e:
        console.print(
            f'[red bold]Error:[/red bold] Report generation failure: {str(e)}'
        )
        raise Exit(code=1)

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)

    console.print(
        f'[bold]Total execution time:[/bold] {int(minutes)} minutes {seconds:.2f} seconds'
    )

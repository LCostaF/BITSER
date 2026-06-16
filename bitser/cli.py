import os
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
from bitser.split_manager import (
    generate_seed,
    make_run_id,
    reconstruct_split,
    split_metadata,
)

app = Typer(
    rich_markup_mode='rich',
    help="""BITSER - Binary Pattern Sequence Recognition

    A command-line tool for training and evaluating ML classifiers on biological sequences
    using k-mer (sliding-window) feature extraction, based on the Local Binary Pattern (LBP) strategy from the field of texture analysis.

    [bold]Workflow:[/bold]
    1. [cyan]metadata[/cyan]  → Parse FASTA sequences and create metadata.tsv (no split)
    2. [cyan]train[/cyan]     → Internally split dataset, run CV + hyperparameter tuning, train final model
    3. [cyan]predict[/cyan]  → Reconstruct the exact test split from stored seed, evaluate on held-out data

    [bold]Commands:[/bold]
    • metadata   Generate metadata.tsv (required first step)
    • train      Train a classification model (XGBoost, Random Forest, SVM, MLP or Naive Bayes)
    • predict    Evaluate the trained model on the held-out test split

    [bold]Examples:[/bold]

    [bold]1. Generate metadata:[/bold]
    bitser metadata -d mydata/ -delim "_"

    [bold]2. Train a model (seed auto-generated if omitted):[/bold]
    bitser train -i mydata/ -dir results/ -o model.pkl --seed 42

    [bold]3. Predict on held-out test split:[/bold]
    bitser predict -m results/model.pkl -dir results/ -d mydata/
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


# ---------------------------------------------------------------------------
# metadata command
# ---------------------------------------------------------------------------


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
    class_which: Annotated[
        int,
        Option(
            '--class-which',
            '-which',
            help='Which occurrence of the delimiter to use (1 = first, -1 = last, default: 1)',
        ),
    ] = 1,
):
    """
    Generate metadata.tsv describing the full dataset.

    Output columns: sample-id, fasta_path, class, record_index.
    No train/test split is written — splitting is performed internally
    by the [cyan]train[/cyan] command using a reproducible stratified split.

    You must specify --class-delim to tell the tool how to find the class label.
    The class token is automatically cleaned to contain only alphanumeric characters.
    """
    if class_which == 0:
        console.print(
            '[red bold]Error:[/red bold] --class-which cannot be 0 (use 1 for first occurrence or -1 for last).'
        )
        raise Exit(code=1)

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
                '[red bold]Error:[/red bold] No valid sequences found after header parsing '
                '(check --class-delim, --class-which, or FASTA headers).'
            )
        elif 'fewer than 2 valid classes' in err_msg:
            console.print(
                '[red bold]Error:[/red bold] Dataset has fewer than 2 valid classes. '
                'Classification requires at least 2 classes.'
            )
        else:
            console.print(f'[red bold]Error:[/red bold] {str(e)}')
        raise Exit(code=1)
    except OSError:
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
                '[red bold]Error:[/red bold] Header parsing issues (delimiter not found in headers, '
                'invalid class-which, or extracted class empty/invalid after cleaning).'
            )
        else:
            console.print(
                f'[red bold]Error:[/red bold] Unexpected error during metadata generation: {str(e)}'
            )
        raise Exit(code=1)

    console.print(f'[bold green]✓ metadata.tsv created at {path}[/bold green]')
    console.print(
        '[dim]Tip: No split has been stored. '
        'Run [cyan]bitser train[/cyan] with [cyan]--seed[/cyan] to control the 80/20 stratified split.[/dim]'
    )


# ---------------------------------------------------------------------------
# train command
# ---------------------------------------------------------------------------


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
    ],
    classifier: Annotated[
        str,
        Option(
            '--classifier',
            '-c',
            help='Classifier algorithm: "rf" (Random Forest), "xgb" (XGBoost), "svm", "mlp", "nb"',
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
            help='Number of cross-validation folds (default: 5)',
        ),
    ] = 5,
    repeats: Annotated[
        int,
        Option(
            '--repeats',
            '-r',
            help='Cross-validation repetitions for variance estimation (default: 1)',
        ),
    ] = 1,
    test_size: Annotated[
        float,
        Option(
            '--test-size',
            help='Fraction of data held out for testing (default: 0.20)',
        ),
    ] = 0.20,
    seed: Annotated[
        int,
        Option(
            '--seed',
            help='Random seed for splitting, CV, and model training. '
            'Auto-generated and logged if not provided.',
        ),
    ] = None,
):
    """
    Train a classification model from sequence data.

    The training process:
    1. Load full dataset from metadata.tsv (no pre-existing split required).
    2. Generate or use the provided --seed; log it for reproducibility.
    3. Perform a stratified 80/20 split (controlled by --seed).
    4. On the training subset only: run stratified k-fold CV (AUROC-based)
       and select best hyperparameters via inner GridSearchCV.
    5. Re-train the final model on the full training subset.
    6. Persist the model together with split definition and run metadata.

    The test subset is NEVER seen during steps 3-5.
    """
    output_path = os.path.join(output_dir, output)

    # -- Validation ----------------------------------------------------------
    if classifier not in {'xgb', 'rf', 'svm', 'mlp', 'nb'}:
        console.print(
            f'[red bold]Error:[/red bold] Unsupported classifier "{classifier}". '
            'Must be one of: xgb, rf, svm, mlp, nb.'
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
    if not (0.0 < test_size < 1.0):
        console.print(
            '[red bold]Error:[/red bold] --test-size must be between 0 and 1 (exclusive).'
        )
        raise Exit(code=1)

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

    # -- Seed ----------------------------------------------------------------
    if seed is None:
        seed = generate_seed()
        console.print(
            f'[yellow]No --seed provided. Auto-generated seed: [bold]{seed}[/bold] '
            f'(logged in model artefact)[/yellow]'
        )
    else:
        console.print(f'  seed        : [yellow]{seed}[/yellow]')

    run_id = make_run_id(seed)
    console.print(f'  run_id      : [yellow]{run_id}[/yellow]')
    console.print(f'  test_size   : [yellow]{test_size}[/yellow]')

    start_time = time.time()
    console.print(
        f'[bold]Training model with {classifier} classifier...[/bold]'
    )

    try:
        # -- Load full metadata ----------------------------------------------
        metadata_path = input_path / 'metadata.tsv'
        full_metadata = pd.read_csv(metadata_path, sep='\t', comment='#')

        # -- Stratified split ------------------------------------------------
        console.print('[cyan]Performing stratified train/test split...[/cyan]')
        train_meta, test_meta, split_def = split_metadata(
            full_metadata, test_size=test_size, seed=seed
        )
        console.print(
            f'  [green]train samples: {len(train_meta)}  |  test samples: {len(test_meta)}[/green]'
        )

        # -- Feature extraction (training subset only) -----------------------
        console.print('[cyan]Extracting features (training subset)...[/cyan]')
        train_features, _, _ = extract_features_from_metadata(
            input,
            metadata_subset=train_meta,
            flank=flank,
            translate_sequences=translate,
        )
        console.print(
            '[bold green]✓ Feature extraction complete![/bold green]'
        )

        console.print('[cyan]Preparing dataframe...[/cyan]')
        train_df, train_classes, name_class = prepare_dataframe(train_features)
        console.print('[bold green]✓ Dataframe prepared![/bold green]')

        # -- CV + hyperparameter tuning + final training ---------------------
        console.print(
            '[cyan]Training model (CV + hyperparameter tuning)...[/cyan]'
        )
        (
            classifier_model,
            min_max_scaler,
            label_encoder,
            _,
            output_text,
            cv_results,
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

        if cv_results.get('cv_auroc_mean') is not None:
            console.print(
                f'  CV AUROC: [bold]{cv_results["cv_auroc_mean"]:.4f}[/bold] '
                f'± {cv_results["cv_auroc_std"]:.4f}'
            )
        if cv_results.get('best_params'):
            console.print(
                f'  Best params: [yellow]{cv_results["best_params"]}[/yellow]'
            )

        save_output_to_file(output_text, classifier, output_dir)

        # -- Persist model + run metadata ------------------------------------
        os.makedirs(output_dir, exist_ok=True)
        save_model(
            classifier_model,
            min_max_scaler,
            label_encoder,
            split_def=split_def,
            output_path=output_path,
            name_class=name_class,
            output_text=output_text,
            train_df_columns=train_df.columns.tolist(),
            cv_results=cv_results,
        )

        console.print(
            f'[bold green]✓ Success![/bold green] Model saved to [cyan]{output_path}[/cyan]'
        )
        console.print(
            f'  Run ID [cyan]{run_id}[/cyan] stored in model — use it with [cyan]bitser predict[/cyan].'
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
        elif 'stratif' in msg or 'split' in msg:
            console.print(
                f'[red bold]Error:[/red bold] Stratified split failed: {str(e)}'
            )
        else:
            console.print(f'[red bold]Error:[/red bold] {str(e)}')
        raise Exit(code=1)
    except OSError:
        console.print(
            '[red bold]Error:[/red bold] Failure writing model file or training output logs '
            '(permissions or disk issues).'
        )
        raise Exit(code=1)
    except Exception as e:
        msg = str(e).lower()
        if 'xgboost' in msg or 'not installed' in msg or 'import' in msg:
            console.print(
                '[red bold]Error:[/red bold] Required library not installed '
                '(e.g., XGBoost for xgb classifier).'
            )
        elif 'cross-validation' in msg or 'splits' in msg or 'samples' in msg:
            console.print(
                '[red bold]Error:[/red bold] Cross-validation infeasible '
                '(too few samples per class for the number of splits/repeats).'
            )
        elif 'feature' in msg or 'columns' in msg or 'flank' in msg:
            console.print(
                '[red bold]Error:[/red bold] Feature extraction or data preparation failure '
                '(inconsistent feature vectors).'
            )
        else:
            console.print(
                f'[red bold]Error:[/red bold] Model training failure: {str(e)}'
            )
        raise Exit(code=1)

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    console.print(
        f'[bold]Total execution time:[/bold] {int(minutes)} minutes {seconds:.2f} seconds'
    )


# ---------------------------------------------------------------------------
# predict command
# ---------------------------------------------------------------------------


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
    ],
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
    Evaluate a trained model on the held-out test split.

    The test split is reconstructed exactly from the seed and split definition
    stored inside the model file — guaranteeing zero leakage from training.

    Output includes:
    - Test AUROC and per-class metrics
    - Confusion matrix
    - Per-sample predictions CSV
    - Reference to the run ID
    """
    test_headers = None
    test_sequences = None

    if flank <= 0:
        console.print(
            '[red bold]Error:[/red bold] --flank must be a positive integer (> 0).'
        )
        raise Exit(code=1)

    start_time = time.time()

    # -- Load model ----------------------------------------------------------
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
    except Exception:
        console.print(
            '[red bold]Error:[/red bold] Model file is corrupted, incompatible, or missing expected keys.'
        )
        raise Exit(code=1)

    # Retrieve persisted run metadata
    split_def = model_data.get('split_def')
    if not split_def:
        console.print(
            '[red bold]Error:[/red bold] Model file does not contain a split definition. '
            'Re-train with the current version of BITSER.'
        )
        raise Exit(code=1)

    run_id = split_def.get('run_id', '<unknown>')
    seed = split_def.get('seed', '<unknown>')
    console.print(f'  run_id : [yellow]{run_id}[/yellow]')
    console.print(f'  seed   : [yellow]{seed}[/yellow]')

    # -- Load full dataset and reconstruct test split ------------------------
    data_path = Path(data).resolve()
    if not data_path.exists():
        console.print(
            f'[red bold]Error:[/red bold] Dataset directory does not exist: [yellow]{data}[/yellow]'
        )
        raise Exit(code=1)
    if not data_path.is_dir():
        console.print(
            f'[red bold]Error:[/red bold] Path is not a directory: [yellow]{data}[/yellow]'
        )
        raise Exit(code=1)
    if not (data_path / 'metadata.tsv').is_file():
        console.print(
            '[red bold]Error:[/red bold] metadata.tsv is missing in the dataset directory.'
        )
        raise Exit(code=1)
    if not (data_path / 'sequences').is_dir():
        console.print(
            '[red bold]Error:[/red bold] sequences/ folder not found in dataset directory.'
        )
        raise Exit(code=1)

    try:
        full_metadata = pd.read_csv(
            data_path / 'metadata.tsv', sep='\t', comment='#'
        )

        console.print('[cyan]Reconstructing held-out test split...[/cyan]')
        _, test_meta = reconstruct_split(full_metadata, split_def)
        console.print(
            f'  [green]test samples: {len(test_meta)}[/green] '
            f'(run_id: {run_id})'
        )
    except KeyError as e:
        console.print(
            f'[red bold]Error:[/red bold] Split reconstruction failed: {str(e)}'
        )
        raise Exit(code=1)
    except Exception as e:
        console.print(
            f'[red bold]Error:[/red bold] Failed to load or reconstruct split: {str(e)}'
        )
        raise Exit(code=1)

    # -- Feature extraction (test subset only) -------------------------------
    console.print('[cyan]Processing test sequences...[/cyan]')
    try:
        (
            test_features,
            test_headers,
            test_sequences,
        ) = extract_features_from_metadata(
            data,
            metadata_subset=test_meta,
            flank=flank,
            translate_sequences=translate,
        )
        test_df, test_classes, _ = prepare_dataframe(test_features)
        console.print('[bold green]✓ Test sequences processed![/bold green]')
    except FileNotFoundError:
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

    classifier_type = type(model_data['classifier']).__name__.lower()

    # -- Predict and evaluate ------------------------------------------------
    console.print('[cyan]Running predictions on held-out test split...[/cyan]')
    try:
        (
            _,
            _,
            _,
            complete_output,
            predictions,
            y_test,
            test_auroc,
        ) = predict_and_evaluate(
            model_data['classifier'],
            model_data['scaler'],
            model_data['encoder'],
            test_df,
            test_classes,
            model_data.get('name_class', []),
            output_dir,
            run_id=run_id,
            train_df=None,
            previous_output=model_data.get('output_text', ''),
            classifier_type=classifier_type,
            validation_df=None,
            validation_classes=None,
        )
        console.print('[bold green]✓ Prediction complete![/bold green]')
        if test_auroc is not None:
            console.print(
                f'  Test AUROC: [bold green]{test_auroc:.4f}[/bold green]'
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
                '[red bold]Error:[/red bold] Feature mismatch with trained model '
                '(different flank or feature columns).'
            )
        else:
            console.print(
                f'[red bold]Error:[/red bold] Prediction execution failure: {str(e)}'
            )
        raise Exit(code=1)

    # -- Save prediction report ----------------------------------------------
    console.print('[cyan]Generating prediction report...[/cyan]')
    try:
        report_df = pd.DataFrame(
            {
                'run_id': run_id,
                'True Class': test_classes.values
                if hasattr(test_classes, 'values')
                else test_classes,
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
    except OSError:
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

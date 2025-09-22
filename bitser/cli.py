import time

import pandas as pd
import pytest
from rich.console import Console
from rich.progress import track
from typer import Context, Exit, Option, Typer
from typing_extensions import Annotated

from bitser import __version__
from bitser.data_preprocessing import prepare_dataframe
from bitser.feature_extraction import extract_features_from_path
from bitser.file_utils import save_output_to_file, save_prediction_report
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
def train(
    input: Annotated[
        str,
        Option(
            '--input',
            '-i',
            help='Directory containing training FASTA files (e.g., "./training_data/")',
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
    start_time = time.time()
    console.print(
        f'[bold]Training model with {classifier} classifier...[/bold]'
    )

    # Extract features with progress indication
    console.print('[cyan]Extracting features...[/cyan]')
    train_features, _, _ = extract_features_from_path(input, flank, translate)
    console.print('[bold green]✓ Feature extraction complete![/bold green]')

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

    save_output_to_file(output_text, classifier)

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

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)

    console.print(
        f'[bold]Total execution time:[/bold] {int(minutes)} minutes {seconds:.2f} seconds'
    )


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
    data: Annotated[
        str,
        Option(
            '--data',
            '-d',
            help='Directory containing test FASTA files',
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
    start_time = time.time()
    console.print(f'[bold]Loading model from {model}...[/bold]')
    model_data = load_model(model)
    console.print(f'[bold green]✓ {model} loaded successfully![/bold green]')

    if data:
        console.print('[cyan]Processing test sequences...[/cyan]')
        (
            test_features,
            test_headers,
            test_sequences,
        ) = extract_features_from_path(data, flank, translate)
        test_df, test_classes, _ = prepare_dataframe(test_features)
        console.print('[bold green]✓ Test sequences processed![/bold green]')
    else:
        if 'test_data' not in model_data:
            console.print(
                '[red]Error:[/red] No test data provided and no saved test data found in model!'
            )
            raise Exit(code=1)
        test_df, test_classes = model_data['test_data']

    classifier_type = type(model_data['classifier']).__name__.lower()

    console.print('[cyan]Running predictions...[/cyan]')

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

    # Generate and save CSV report
    console.print('[cyan]Generating prediction report...[/cyan]')
    report_df = pd.DataFrame(
        {
            'True Class': test_classes,  # Original class names
            'Label': test_headers
            if test_headers is not None
            else [None] * len(test_classes),  # From FASTA headers
            'Sequence': test_sequences
            if test_sequences is not None
            else [None] * len(test_classes),  # Actual sequences
            'Predicted Class': predictions,  # Predicted class names
        }
    )

    csv_path = save_prediction_report(report_df, classifier_type)
    console.print(
        f'[bold green]✓ Prediction report saved to {csv_path}![/bold green]'
    )

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)

    console.print(
        f'[bold]Total execution time:[/bold] {int(minutes)} minutes {seconds:.2f} seconds'
    )

import re
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from bitser import __version__
from bitser.cli import app, main

runner = CliRunner()


def test_main_no_command(capsys):
    # Test main function without subcommand
    ctx = MagicMock()
    ctx.invoked_subcommand = None
    main(ctx)
    captured = capsys.readouterr()
    assert 'Welcome to BITSER!' in captured.out


def test_main_with_command():
    # Test main function with subcommand (should do nothing)
    ctx = MagicMock()
    ctx.invoked_subcommand = 'train'
    assert main(ctx) is None


def test_cli_version_flag():
    # Test --version flag
    import re

    result = runner.invoke(app, ['--version'])
    assert result.exit_code == 0

    # Strip ANSI color codes from the output
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean_output = ansi_escape.sub('', result.stdout)

    assert __version__ in clean_output


def test_cli_help():
    # Test help output
    result = runner.invoke(app, ['--help'])
    assert result.exit_code == 0
    assert (
        'BITSER - Bioinformatics Tool for Sequence Classification'
        in result.stdout
    )
    assert 'Train a model:' in result.stdout
    assert 'Test sequences:' in result.stdout


@patch('bitser.cli.extract_features_from_path')
@patch('bitser.cli.prepare_dataframe')
@patch('bitser.cli.train_classification_model')
@patch('bitser.cli.save_output_to_file')
@patch('bitser.cli.save_model')
def test_train_command(
    mock_save_model, mock_save_output, mock_train, mock_prepare, mock_extract
):
    # Mock return values
    mock_extract.return_value = {'features': 'mock'}
    mock_df = MagicMock()
    mock_df.columns.tolist.return_value = [
        'feat1',
        'feat2',
    ]  # Add columns mock
    mock_prepare.return_value = (
        mock_df,
        ['class1', 'class2'],
        {'name': 'class'},
    )
    mock_train.return_value = (
        'classifier',
        'scaler',
        'encoder',
        'cv_results',
        'output_text',
    )

    # Test train command
    result = runner.invoke(
        app, ['train', '--input', 'data/', '--output', 'model.pkl']
    )
    assert result.exit_code == 0

    # Verify mocks
    mock_extract.assert_called_once_with(
        'data/', 8, False
    )  # Check default args
    mock_prepare.assert_called_once_with({'features': 'mock'})
    mock_save_output.assert_called_once_with('output_text', 'xgb')
    mock_save_model.assert_called_once_with(
        'classifier',
        'scaler',
        'encoder',
        None,
        'model.pkl',
        name_class={'name': 'class'},
        output_text='output_text',
        train_df_columns=[
            'feat1',
            'feat2',
        ],  # Use concrete values instead of ANY
    )

    # Test with all args
    result = runner.invoke(
        app,
        [
            'train',
            '--input',
            'data/',
            '--output',
            'model.pkl',
            '--classifier',
            'rf',
            '--flank',
            '5',
            '--translate',
            '--splits',
            '5',
            '--repeats',
            '3',
            '--seed',
            '42',
        ],
    )
    assert result.exit_code == 0
    assert 'rf' in result.stdout

    # Verify classifier type was passed correctly
    mock_save_output.assert_called_with('output_text', 'rf')


@patch('bitser.cli.load_model')
@patch('bitser.cli.extract_features_from_path')
@patch('bitser.cli.prepare_dataframe')
@patch('bitser.cli.predict_and_evaluate')
def test_predict_command(mock_predict, mock_prepare, mock_extract, mock_load):
    # Mock return values
    mock_load.return_value = {
        'classifier': 'mock_classifier',
        'scaler': 'mock_scaler',
        'encoder': 'mock_encoder',
        'name_class': ['class1', 'class2'],
        'output_text': 'previous_output',
    }
    mock_extract.return_value = {'features': 'mock'}
    mock_prepare.return_value = (MagicMock(), ['class1', 'class2'], {})
    mock_predict.return_value = (None, None, None, 'complete_output')

    # Test predict command with test data
    result = runner.invoke(
        app,
        [
            'predict',
            '--model',
            'model.pkl',
            '--data',
            'test_data/',
        ],
    )
    assert result.exit_code == 0
    assert 'Loading model from model.pkl' in result.stdout

    # Test predict command without test data (should fail)
    mock_load.return_value = {}  # No test_data in model
    result = runner.invoke(
        app,
        [
            'predict',
            '--model',
            'model.pkl',
        ],
    )
    assert result.exit_code == 1
    assert 'Error:' in result.stdout


@patch('bitser.cli.load_model')
@patch('bitser.cli.predict_and_evaluate')
def test_predict_command_with_saved_test_data(mock_predict, mock_load):
    # Mock the model with test_data included
    mock_test_df = MagicMock()
    mock_test_classes = ['class1', 'class2']
    mock_model_data = {
        'classifier': MagicMock(),
        'scaler': 'mock_scaler',
        'encoder': 'mock_encoder',
        'name_class': ['class1', 'class2'],
        'output_text': 'previous_output',
        'test_data': (
            mock_test_df,
            mock_test_classes,
        ),  # This is the key addition
    }
    mock_load.return_value = mock_model_data

    # Mock the return value from predict_and_evaluate
    mock_predict.return_value = (None, None, None, 'complete_output')

    # Get type of classifier for classifier_type determination
    mock_model_data['classifier'].__class__.__name__ = 'RandomForestClassifier'

    # Test predict command without providing test data (should use saved test data)
    result = runner.invoke(
        app,
        [
            'predict',
            '--model',
            'model.pkl',
        ],
    )

    # Check that the command succeeded
    assert result.exit_code == 0
    assert 'Loading model from model.pkl' in result.stdout
    assert (
        'Processing test sequences' not in result.stdout
    )  # Should not process new test sequences

    # Verify predict_and_evaluate was called with the correct arguments from the saved test data
    mock_predict.assert_called_once_with(
        mock_model_data['classifier'],
        mock_model_data['scaler'],
        mock_model_data['encoder'],
        mock_test_df,
        mock_test_classes,
        mock_model_data['name_class'],
        train_df=None,
        previous_output=mock_model_data['output_text'],
        classifier_type='randomforestclassifier',  # lowercase of the class name
        validation_df=None,
        validation_classes=None,
        save_files=True,
    )


def test_invalid_command():
    # Test invalid command
    result = runner.invoke(app, ['invalid-command'])
    assert result.exit_code != 0
    assert 'No such command' in result.stdout

import os
from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from bitser import file_utils


# Create a temporary directory for testing
@pytest.fixture
def temp_dir(tmpdir):
    return tmpdir


def test_get_project_dir():
    # Test that get_project_dir returns a string and a valid directory
    project_dir = file_utils.get_project_dir()
    assert isinstance(project_dir, str)
    assert os.path.isdir(project_dir)


# Mock directory functions to use the temporary directory
@pytest.fixture
def mock_dirname(monkeypatch, temp_dir):
    # Mock os.path.dirname to return temp_dir regardless of input
    def mock_dirname_func(path):
        return str(temp_dir)

    monkeypatch.setattr(os.path, 'dirname', mock_dirname_func)
    return temp_dir


def test_create_results_directory(mock_dirname, temp_dir):
    # Define the expected results directory
    results_dir = os.path.join(temp_dir, 'results')

    # Make sure it doesn't exist before the test
    if os.path.exists(results_dir):
        os.rmdir(results_dir)

    # Call the function
    created_dir = file_utils.create_results_directory()

    # Check if the directory was created
    assert os.path.exists(results_dir)
    assert os.path.isdir(results_dir)
    assert created_dir == results_dir


def test_create_validation_directory(mock_dirname, temp_dir):
    # Define the expected validation directory
    validation_dir = os.path.join(temp_dir, 'validation_data')

    # Make sure it doesn't exist before the test
    if os.path.exists(validation_dir):
        os.rmdir(validation_dir)

    # Call the function
    created_dir = file_utils.create_validation_directory()

    # Check if the directory was created
    assert os.path.exists(validation_dir)
    assert os.path.isdir(validation_dir)
    assert created_dir == validation_dir


def test_save_output_to_file(mock_dirname, temp_dir):
    # Test data
    output_text = 'Test output text'
    classifier_type = 'rf'

    # Mock datetime to get a predictable filename
    fixed_datetime = datetime(2023, 1, 1, 12, 0, 0)
    datetime_str = fixed_datetime.strftime('%Y%m%d_%H%M%S')
    expected_filename = f'{datetime_str}_results_{classifier_type}.txt'

    with patch('bitser.file_utils.datetime') as mock_datetime:
        mock_datetime.now.return_value = fixed_datetime

        # Call the function
        file_path = file_utils.save_output_to_file(
            output_text, classifier_type
        )

    # Expected file path
    results_dir = os.path.join(temp_dir, 'results')
    expected_file_path = os.path.join(results_dir, expected_filename)

    # Check if the file was created with the right path
    assert file_path == expected_file_path
    assert os.path.exists(file_path)

    # Check the content of the file
    with open(file_path, 'r') as f:
        content = f.read()
    assert content == output_text


def test_save_validation_data(mock_dirname, temp_dir):
    # Test data
    validation_df = pd.DataFrame(
        {'feature_1': [1, 2, 3], 'feature_2': [4, 5, 6]}
    )
    validation_classes = pd.Series([0, 1, 0], name='class')
    classifier_type = 'xgb'

    # Mock datetime to get a predictable filename
    fixed_datetime = datetime(2023, 1, 1, 12, 0, 0)
    datetime_str = fixed_datetime.strftime('%Y%m%d_%H%M%S')
    expected_filename = f'{datetime_str}_validation_data_{classifier_type}.csv'

    with patch('bitser.file_utils.datetime') as mock_datetime:
        mock_datetime.now.return_value = fixed_datetime

        # Call the function
        file_utils.save_validation_data(
            validation_df, validation_classes, classifier_type
        )

    # Expected file path
    validation_dir = os.path.join(temp_dir, 'validation_data')
    expected_file_path = os.path.join(validation_dir, expected_filename)

    # Check if the file was created
    assert os.path.exists(expected_file_path)

    # Check the content of the file
    loaded_data = pd.read_csv(expected_file_path)
    assert list(loaded_data.columns) == ['feature_1', 'feature_2', 'class']
    assert len(loaded_data) == 3
    assert loaded_data['class'].tolist() == [0, 1, 0]
    assert loaded_data['feature_1'].tolist() == [1, 2, 3]
    assert loaded_data['feature_2'].tolist() == [4, 5, 6]


# Test for directory existence scenarios
def test_create_results_directory_existing(mock_dirname, temp_dir):
    # Create the directory before calling the function
    results_dir = os.path.join(temp_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Call the function
    created_dir = file_utils.create_results_directory()

    # Check the result
    assert created_dir == results_dir
    assert os.path.exists(results_dir)


def test_create_validation_directory_existing(mock_dirname, temp_dir):
    # Create the directory before calling the function
    validation_dir = os.path.join(temp_dir, 'validation_data')
    os.makedirs(validation_dir, exist_ok=True)

    # Call the function
    created_dir = file_utils.create_validation_directory()

    # Check the result
    assert created_dir == validation_dir
    assert os.path.exists(validation_dir)

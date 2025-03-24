import os
import pytest
import pandas as pd
from datetime import datetime
from bitser import file_utils


# Create a temporary directory for testing
@pytest.fixture
def temp_dir(tmpdir):
    return tmpdir


# Mock the project directory to be the temporary directory
@pytest.fixture
def mock_project_dir(monkeypatch, temp_dir):
    monkeypatch.setattr(file_utils, "get_project_dir", lambda: str(temp_dir))


def test_create_results_directory(mock_project_dir, temp_dir):
    results_dir = os.path.join(temp_dir, "results")

    # Call the function
    created_dir = file_utils.create_results_directory()

    # Check if the directory was created
    assert os.path.exists(results_dir)
    assert created_dir == str(results_dir)


def test_create_validation_directory(mock_project_dir, temp_dir):
    validation_dir = os.path.join(temp_dir, "validation_data")

    # Call the function
    created_dir = file_utils.create_validation_directory()

    # Check if the directory was created
    assert os.path.exists(validation_dir)
    assert created_dir == str(validation_dir)


def test_save_output_to_file(mock_project_dir, temp_dir):
    # Test data
    output_text = "Test output text"
    classifier_type = "rf"

    # Call the function
    file_path = file_utils.save_output_to_file(output_text, classifier_type)

    # Check if the file was created
    assert os.path.exists(file_path)

    # Check the content of the file
    with open(file_path, 'r') as f:
        content = f.read()
    assert content == output_text

    # Check the filename format
    assert classifier_type in file_path
    assert datetime.now().strftime("%Y%m%d") in file_path


def test_save_validation_data(mock_project_dir, temp_dir):
    # Test data
    validation_df = pd.DataFrame({
        "feature_1": [1, 2, 3],
        "feature_2": [4, 5, 6]
    })
    validation_classes = pd.Series([0, 1, 0], name="class")
    classifier_type = "xgb"

    # Call the function
    file_utils.save_validation_data(validation_df, validation_classes, classifier_type)

    # Check if the file was created
    validation_dir = os.path.join(temp_dir, "validation_data")
    files = os.listdir(validation_dir)
    assert len(files) == 1

    # Check the filename format
    file_path = os.path.join(validation_dir, files[0])
    assert classifier_type in file_path
    assert datetime.now().strftime("%Y%m%d") in file_path

    # Check the content of the file
    loaded_data = pd.read_csv(file_path)
    assert "class" in loaded_data.columns
    assert len(loaded_data) == 3

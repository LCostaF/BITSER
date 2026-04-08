import os
from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from bitser import file_utils


@pytest.fixture
def temp_dir(tmpdir):
    return tmpdir


# ==================== save_output_to_file ====================


def test_save_output_to_file(temp_dir):
    output_text = 'Test output text'
    classifier_type = 'rf'
    output_dir = os.path.join(str(temp_dir), 'results')

    # Mock datetime for predictable filename
    fixed_datetime = datetime(2023, 1, 1, 12, 0, 0)
    datetime_str = fixed_datetime.strftime('%Y%m%d_%H%M%S')
    expected_filename = f'{datetime_str}_results_{classifier_type}.txt'
    expected_file_path = os.path.join(output_dir, expected_filename)

    with patch('bitser.file_utils.datetime') as mock_datetime:
        mock_datetime.now.return_value = fixed_datetime

        file_path = file_utils.save_output_to_file(
            output_text, classifier_type, output_dir
        )

    assert file_path == expected_file_path
    assert os.path.exists(file_path)

    with open(file_path, 'r') as f:
        content = f.read()
    assert content == output_text


def test_save_output_to_file_raises_when_no_output_dir():
    with pytest.raises(ValueError, match='output_dir must be provided'):
        file_utils.save_output_to_file('some text', 'rf', None)


# ==================== save_validation_data ====================


def test_save_validation_data(temp_dir):
    validation_df = pd.DataFrame(
        {'feature_1': [1, 2, 3], 'feature_2': [4, 5, 6]}
    )
    validation_classes = pd.Series([0, 1, 0], name='class')
    classifier_type = 'xgb'
    output_dir = os.path.join(str(temp_dir), 'validation_data')

    fixed_datetime = datetime(2023, 1, 1, 12, 0, 0)
    datetime_str = fixed_datetime.strftime('%Y%m%d_%H%M%S')
    expected_filename = f'{datetime_str}_validation_data_{classifier_type}.csv'
    expected_file_path = os.path.join(output_dir, expected_filename)

    with patch('bitser.file_utils.datetime') as mock_datetime:
        mock_datetime.now.return_value = fixed_datetime

        file_utils.save_validation_data(
            validation_df, validation_classes, classifier_type, output_dir
        )

    assert os.path.exists(expected_file_path)

    loaded_data = pd.read_csv(expected_file_path)
    assert list(loaded_data.columns) == ['feature_1', 'feature_2', 'class']
    assert len(loaded_data) == 3
    assert loaded_data['class'].tolist() == [0, 1, 0]
    assert loaded_data['feature_1'].tolist() == [1, 2, 3]
    assert loaded_data['feature_2'].tolist() == [4, 5, 6]


def test_save_validation_data_raises_when_no_output_dir():
    df = pd.DataFrame({'a': [1]})
    with pytest.raises(ValueError, match='output_dir must be provided'):
        file_utils.save_validation_data(df, pd.Series([0]), 'xgb', None)


# ==================== save_prediction_report ====================


def test_save_prediction_report(temp_dir):
    report_df = pd.DataFrame(
        {
            'id': [1, 2, 3],
            'prediction': [0.8, 0.3, 0.9],
            'true_label': [1, 0, 1],
        }
    )
    classifier_type = 'rf'
    output_dir = os.path.join(str(temp_dir), 'predictions')

    fixed_datetime = datetime(2023, 1, 1, 12, 0, 0)
    datetime_str = fixed_datetime.strftime('%Y%m%d_%H%M%S')
    expected_filename = f'{datetime_str}_{classifier_type}_predictions.csv'
    expected_file_path = os.path.join(output_dir, expected_filename)

    with patch('bitser.file_utils.datetime') as mock_datetime:
        mock_datetime.now.return_value = fixed_datetime

        returned_path = file_utils.save_prediction_report(
            report_df, classifier_type, output_dir
        )

    assert returned_path == expected_file_path
    assert os.path.exists(expected_file_path)

    loaded = pd.read_csv(expected_file_path)
    pd.testing.assert_frame_equal(loaded, report_df)


def test_save_prediction_report_raises_when_no_output_dir():
    df = pd.DataFrame({'pred': [0.5]})
    with pytest.raises(ValueError, match='output_dir must be provided'):
        file_utils.save_prediction_report(df, 'rf', None)

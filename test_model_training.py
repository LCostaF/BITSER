import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from bitser import model_training


# Fixture to create a synthetic dataset for testing
@pytest.fixture
def synthetic_data():
    x, y = make_classification(
        n_samples=100,
        n_features=10,
        n_classes=3,
        n_informative=5,
        random_state=42,
    )
    df = pd.DataFrame(x, columns=[f'feature_{i}' for i in range(x.shape[1])])
    classes = pd.Series(y, name='class')
    return df, classes


# Fixture for small dataset with imbalanced classes
@pytest.fixture
def small_imbalanced_data():
    # Create a dataset with one class having only 1 sample
    x = np.array(
        [
            [0.1, 0.2, 0.3],  # Class 0
            [0.4, 0.5, 0.6],  # Class 1
            [0.7, 0.8, 0.9],  # Class 1
            [1.0, 1.1, 1.2],  # Class 2
            [1.3, 1.4, 1.5],  # Class 2
        ]
    )
    y = np.array([0, 1, 1, 2, 2])  # Class 0 has only 1 sample

    df = pd.DataFrame(x, columns=[f'feature_{i}' for i in range(x.shape[1])])
    classes = pd.Series(y, name='class')
    return df, classes


# Fixture for small dataset with few samples per class
@pytest.fixture
def small_few_samples_data():
    # Create a dataset with classes having fewer samples than default n_splits
    x = np.array(
        [
            [0.1, 0.2, 0.3],  # Class 0
            [0.4, 0.5, 0.6],  # Class 0
            [0.7, 0.8, 0.9],  # Class 1
            [1.0, 1.1, 1.2],  # Class 1
            [1.3, 1.4, 1.5],  # Class 2
            [1.6, 1.7, 1.8],  # Class 2
        ]
    )
    y = np.array([0, 0, 1, 1, 2, 2])  # Each class has only 2 samples

    df = pd.DataFrame(x, columns=[f'feature_{i}' for i in range(x.shape[1])])
    classes = pd.Series(y, name='class')
    return df, classes


def test_train_classification_model(synthetic_data):
    train_df, train_classes = synthetic_data

    # Test RandomForestClassifier
    (
        classifier,
        min_max_scaler,
        label_encoder,
        train_df_scaled,
        output_text,
    ) = model_training.train_classification_model(
        train_df, train_classes, classifier_type='rf', perform_cv=False
    )
    assert isinstance(classifier, RandomForestClassifier)
    assert isinstance(min_max_scaler, MinMaxScaler)
    assert isinstance(label_encoder, LabelEncoder)
    assert train_df_scaled.shape == train_df.shape
    assert isinstance(output_text, str)  # Check that output_text is a string

    # Test XGBoost
    (
        classifier,
        min_max_scaler,
        label_encoder,
        train_df_scaled,
        output_text,
    ) = model_training.train_classification_model(
        train_df, train_classes, classifier_type='xgb', perform_cv=False
    )
    assert isinstance(classifier, XGBClassifier)
    assert isinstance(min_max_scaler, MinMaxScaler)
    assert isinstance(label_encoder, LabelEncoder)
    assert train_df_scaled.shape == train_df.shape
    assert isinstance(output_text, str)  # Check that output_text is a string

    # Test SVC classifier
    (
        classifier,
        min_max_scaler,
        label_encoder,
        train_df_scaled,
        output_text,
    ) = model_training.train_classification_model(
        train_df, train_classes, classifier_type='svm', perform_cv=False
    )
    assert isinstance(classifier, SVC)
    assert isinstance(min_max_scaler, MinMaxScaler)
    assert isinstance(label_encoder, LabelEncoder)
    assert train_df_scaled.shape == train_df.shape
    assert isinstance(output_text, str)

    # Test MLPClassifier
    (
        classifier,
        min_max_scaler,
        label_encoder,
        train_df_scaled,
        output_text,
    ) = model_training.train_classification_model(
        train_df, train_classes, classifier_type='mlp', perform_cv=False
    )
    assert isinstance(classifier, MLPClassifier)
    assert isinstance(min_max_scaler, MinMaxScaler)
    assert isinstance(label_encoder, LabelEncoder)
    assert train_df_scaled.shape == train_df.shape
    assert isinstance(output_text, str)

    # Test GaussianNB
    (
        classifier,
        min_max_scaler,
        label_encoder,
        train_df_scaled,
        output_text,
    ) = model_training.train_classification_model(
        train_df, train_classes, classifier_type='nb', perform_cv=False
    )
    assert isinstance(classifier, GaussianNB)
    assert isinstance(min_max_scaler, MinMaxScaler)
    assert isinstance(label_encoder, LabelEncoder)
    assert train_df_scaled.shape == train_df.shape
    assert isinstance(output_text, str)

    # Test unsupported classifier type
    with pytest.raises(ValueError):
        model_training.train_classification_model(
            train_df,
            train_classes,
            classifier_type='unsupported',
            perform_cv=False,
        )


def test_cross_validation_full_branch():
    # Create a dataset large enough for normal CV path (no warnings)
    X, y = make_classification(
        n_samples=60,
        n_features=5,
        n_informative=3,
        n_classes=3,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
    classes = pd.Series(y)

    (
        classifier,
        min_max_scaler,
        label_encoder,
        train_df_scaled,
        output_text,
    ) = model_training.train_classification_model(
        df,
        classes,
        classifier_type='rf',
        perform_cv=True,
        n_splits=3,
        n_repeats=2,
        seed=42,
    )

    # Check that the cross-validation results were printed
    assert 'Cross-validation scores:' in output_text
    assert 'accuracy:' in output_text
    assert 'precision_macro:' in output_text
    assert 'recall_macro:' in output_text
    assert 'f1_macro:' in output_text

    # Check that the classifier was trained successfully
    assert isinstance(classifier, RandomForestClassifier)
    assert not train_df_scaled.empty


def test_cross_validation_single_sample_class(small_imbalanced_data):
    """Test cross-validation with a class having only 1 sample."""
    train_df, train_classes = small_imbalanced_data

    # Test with cross-validation
    (
        classifier,
        min_max_scaler,
        label_encoder,
        train_df_scaled,
        output_text,
    ) = model_training.train_classification_model(
        train_df,
        train_classes,
        classifier_type='rf',
        perform_cv=True,
        n_splits=5,
        n_repeats=2,
    )

    # Check that the warning message is present in the output
    assert (
        'Cannot perform k-fold cross-validation because smallest class has too few samples'
        in output_text
    )
    assert 'Skipping cross-validation' in output_text

    # Check that the model was still trained successfully
    assert isinstance(classifier, RandomForestClassifier)


def test_cross_validation_few_samples_per_class(small_few_samples_data):
    """Test cross-validation with classes having fewer samples than n_splits."""
    train_df, train_classes = small_few_samples_data

    # Each class has 2 samples, so with n_splits=5, it should reduce to 2
    (
        classifier,
        min_max_scaler,
        label_encoder,
        train_df_scaled,
        output_text,
    ) = model_training.train_classification_model(
        train_df,
        train_classes,
        classifier_type='rf',
        perform_cv=True,
        n_splits=5,
        n_repeats=2,
    )

    # Check that the warning message about reducing splits is present
    assert 'Reducing n_splits from 5 to 2' in output_text

    # Check for cross-validation scores in the output
    assert 'Cross-validation scores:' in output_text
    assert 'accuracy:' in output_text
    assert 'precision_macro:' in output_text
    assert 'recall_macro:' in output_text
    assert 'f1_macro:' in output_text

    # Check that the model was trained successfully
    assert isinstance(classifier, RandomForestClassifier)


def test_predict_and_evaluate(synthetic_data):
    train_df, train_classes = synthetic_data
    test_df, test_classes = synthetic_data

    (
        classifier,
        min_max_scaler,
        label_encoder,
        train_df_scaled,
        output_text,
    ) = model_training.train_classification_model(
        train_df, train_classes, classifier_type='rf', perform_cv=False
    )

    name_class = [str(cls) for cls in label_encoder.classes_]

    # Test with basic parameters
    (
        classifier,
        min_max_scaler,
        label_encoder,
        complete_output,
        predictions,
        y_test,
    ) = model_training.predict_and_evaluate(
        classifier,
        min_max_scaler,
        label_encoder,
        test_df,
        test_classes,
        name_class=name_class,
        save_files=False,
    )

    # Check if the returns are correct
    assert isinstance(classifier, RandomForestClassifier)
    assert isinstance(min_max_scaler, MinMaxScaler)
    assert isinstance(label_encoder, LabelEncoder)
    assert isinstance(complete_output, str)

    # Test with validation data and previous output
    validation_df, validation_classes = synthetic_data
    previous_output = 'Previous output text'

    (
        classifier,
        min_max_scaler,
        label_encoder,
        complete_output,
        predictions,
        y_test,
    ) = model_training.predict_and_evaluate(
        classifier,
        min_max_scaler,
        label_encoder,
        test_df,
        test_classes,
        name_class=name_class,
        train_df=train_df,
        previous_output=previous_output,
        classifier_type='rf',
        validation_df=validation_df,
        validation_classes=validation_classes,
        save_files=False,
    )

    # Check if previous output is included
    assert previous_output in complete_output
    assert isinstance(complete_output, str)


def test_predict_and_evaluate_with_save_files(synthetic_data):
    """Test save_files parameter in predict_and_evaluate function."""
    train_df, train_classes = synthetic_data
    test_df, test_classes = synthetic_data

    (
        classifier,
        min_max_scaler,
        label_encoder,
        _,
        _,
    ) = model_training.train_classification_model(
        train_df, train_classes, classifier_type='rf', perform_cv=False
    )

    name_class = [str(cls) for cls in label_encoder.classes_]

    # Mock the save_output_to_file function to check if it's called
    with patch('bitser.model_training.save_output_to_file') as mock_save:
        (
            classifier,
            min_max_scaler,
            label_encoder,
            complete_output,
            predictions,
            y_test,
        ) = model_training.predict_and_evaluate(
            classifier,
            min_max_scaler,
            label_encoder,
            test_df,
            test_classes,
            name_class=name_class,
            classifier_type='rf',
            save_files=True,  # Enable saving
        )

        # Check if save_output_to_file was called with the correct parameters
        mock_save.assert_called_once_with(complete_output, 'rf')


def test_save_and_load_model(synthetic_data):
    train_df, train_classes = synthetic_data
    test_df, test_classes = synthetic_data

    # Train model
    (
        classifier,
        min_max_scaler,
        label_encoder,
        _,
        output_text,
    ) = model_training.train_classification_model(
        train_df, train_classes, classifier_type='rf', perform_cv=False
    )

    # Create temporary file for model
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        model_path = tmp.name

    try:
        # Test saving model with test data
        name_class = [str(cls) for cls in label_encoder.classes_]
        model_training.save_model(
            classifier,
            min_max_scaler,
            label_encoder,
            test_data=(test_df, test_classes),
            output_path=model_path,
            name_class=name_class,
            output_text=output_text,
            train_df_columns=train_df.columns,
        )

        # Test loading model
        loaded_data = model_training.load_model(model_path)

        # Verify contents of loaded data
        assert isinstance(loaded_data, dict)
        assert isinstance(loaded_data['classifier'], RandomForestClassifier)
        assert isinstance(loaded_data['scaler'], MinMaxScaler)
        assert isinstance(loaded_data['encoder'], LabelEncoder)
        assert isinstance(loaded_data['test_data'], tuple)
        assert isinstance(loaded_data['name_class'], list)
        assert isinstance(loaded_data['output_text'], str)
        assert isinstance(loaded_data['train_df_columns'], pd.Index)

    finally:
        # Clean up temporary file
        if os.path.exists(model_path):
            os.unlink(model_path)

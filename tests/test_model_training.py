import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
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

    # Test unsupported classifier type
    with pytest.raises(ValueError):
        model_training.train_classification_model(
            train_df,
            train_classes,
            classifier_type='unsupported',
            perform_cv=False,
        )


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


def test_run_classification(synthetic_data):
    train_df, train_classes = synthetic_data
    test_df, test_classes = synthetic_data

    name_class = [str(cls) for cls in np.unique(train_classes)]

    # Test with test_df provided (train-test mode)
    (
        classifier,
        min_max_scaler,
        label_encoder,
        output_text,
    ) = model_training.run_classification(
        train_df,
        test_df,
        train_classes,
        test_classes,
        name_class=name_class,
        classifier_type='rf',
        save_files=False,
    )
    assert isinstance(classifier, RandomForestClassifier)
    assert isinstance(min_max_scaler, MinMaxScaler)
    assert isinstance(label_encoder, LabelEncoder)
    assert isinstance(output_text, str)

    # Test without test_df (cross-validation mode)
    (
        classifier,
        min_max_scaler,
        label_encoder,
        output_text,
    ) = model_training.run_classification(
        train_df,
        None,
        train_classes,
        None,
        name_class=name_class,
        classifier_type='rf',
        save_files=False,
    )
    assert isinstance(classifier, RandomForestClassifier)
    assert isinstance(min_max_scaler, MinMaxScaler)
    assert isinstance(label_encoder, LabelEncoder)
    assert isinstance(output_text, str)


def test_run_classification_error_handling(synthetic_data):
    train_df, train_classes = synthetic_data

    # Test with invalid classifier type
    with pytest.raises(ValueError):
        model_training.run_classification(
            train_df,
            None,
            train_classes,
            None,
            name_class=np.unique(train_classes),
            classifier_type='unsupported',
            save_files=False,
        )


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


def test_train_and_save(synthetic_data):
    train_df, train_classes = synthetic_data

    # Create temporary file for model
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        model_path = tmp.name

    try:
        # Test train_and_save function
        model_training.train_and_save(
            train_df, train_classes, model_path, classifier_type='rf', seed=7
        )

        # Verify model was saved correctly
        assert os.path.exists(model_path)

        # Test loading the saved model
        loaded_data = model_training.load_model(model_path)
        assert isinstance(loaded_data['classifier'], RandomForestClassifier)
        assert isinstance(loaded_data['scaler'], MinMaxScaler)
        assert isinstance(loaded_data['encoder'], LabelEncoder)
        assert isinstance(loaded_data['test_data'], tuple)

    finally:
        # Clean up temporary file
        if os.path.exists(model_path):
            os.unlink(model_path)

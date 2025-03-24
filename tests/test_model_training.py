import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from bitser import model_training


# Fixture to create a synthetic dataset for testing
@pytest.fixture
def synthetic_data():
    x, y = make_classification(n_samples=100, n_features=10, n_classes=3, n_informative=5, random_state=42)
    df = pd.DataFrame(x, columns=[f"feature_{i}" for i in range(x.shape[1])])
    classes = pd.Series(y, name="class")
    return df, classes


def test_train_classification_model(synthetic_data):
    train_df, train_classes = synthetic_data

    # Test RandomForestClassifier
    classifier, min_max_scaler, label_encoder, train_df_scaled, output_text = model_training.train_classification_model(
        train_df, train_classes, classifier_type='rf', perform_cv=False
    )
    assert isinstance(classifier, RandomForestClassifier)
    assert isinstance(min_max_scaler, MinMaxScaler)
    assert isinstance(label_encoder, LabelEncoder)
    assert train_df_scaled.shape == train_df.shape

    # Test XGBoost
    classifier, min_max_scaler, label_encoder, train_df_scaled, output_text = model_training.train_classification_model(
        train_df, train_classes, classifier_type='xgb', perform_cv=False
    )
    assert isinstance(classifier, XGBClassifier)
    assert isinstance(min_max_scaler, MinMaxScaler)
    assert isinstance(label_encoder, LabelEncoder)
    assert train_df_scaled.shape == train_df.shape

    # Test unsupported classifier type
    with pytest.raises(ValueError):
        model_training.train_classification_model(train_df, train_classes, classifier_type='unsupported', perform_cv=False)


def test_predict_and_evaluate(synthetic_data):
    train_df, train_classes = synthetic_data
    test_df, test_classes = synthetic_data

    classifier, min_max_scaler, label_encoder, train_df_scaled, _ = model_training.train_classification_model(
        train_df, train_classes, classifier_type='rf', perform_cv=False
    )

    name_class = [str(cls) for cls in label_encoder.classes_]

    classifier, min_max_scaler, label_encoder = model_training.predict_and_evaluate(
        classifier, min_max_scaler, label_encoder, test_df, test_classes, name_class=name_class, save_files=False
    )

    # Check if the classifier is returned correctly
    assert isinstance(classifier, RandomForestClassifier)
    assert isinstance(min_max_scaler, MinMaxScaler)
    assert isinstance(label_encoder, LabelEncoder)


def test_run_classification(synthetic_data):
    train_df, train_classes = synthetic_data
    test_df, test_classes = synthetic_data

    name_class = [str(cls) for cls in np.unique(train_classes)]

    # Test with test_df provided (train-test mode)
    classifier, min_max_scaler, label_encoder = model_training.run_classification(
        train_df, test_df, train_classes, test_classes, name_class=name_class, classifier_type='rf', save_files=False
    )
    assert isinstance(classifier, RandomForestClassifier)
    assert isinstance(min_max_scaler, MinMaxScaler)
    assert isinstance(label_encoder, LabelEncoder)

    # Test without test_df (cross-validation mode)
    classifier, min_max_scaler, label_encoder = model_training.run_classification(
        train_df, None, train_classes, None, name_class=name_class, classifier_type='rf', save_files=False
    )
    assert isinstance(classifier, RandomForestClassifier)
    assert isinstance(min_max_scaler, MinMaxScaler)
    assert isinstance(label_encoder, LabelEncoder)


def test_run_classification_error_handling(synthetic_data):
    train_df, train_classes = synthetic_data

    # Test with invalid classifier type
    with pytest.raises(ValueError):
        model_training.run_classification(train_df, None, train_classes, None, name_class=np.unique(train_classes), classifier_type='unsupported', save_files=False)
        
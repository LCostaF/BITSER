"""
tests/test_model_training.py
-----------------------------
Tests for the refactored model_training module.

Key changes from the original:
- train_classification_model now returns 6 values (added cv_results dict).
- save_model stores split_def (replaces test_data).
- predict_and_evaluate returns 7 values (added test_auroc).
- CV uses AUROC as primary metric; hyperparameter tuning runs inside CV.
"""
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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_data():
    X, y_num = make_classification(
        n_samples=120,
        n_features=10,
        n_classes=3,
        n_informative=5,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    classes = pd.Series(['cls' + str(v) for v in y_num], name='class')
    return df, classes


@pytest.fixture
def small_imbalanced_data():
    X = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5],
        ]
    )
    y = ['A', 'B', 'B', 'C', 'C']  # A has only 1 sample
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(3)])
    classes = pd.Series(y, name='class')
    return df, classes


@pytest.fixture
def small_few_samples_data():
    X = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5],
            [1.6, 1.7, 1.8],
        ]
    )
    y = ['A', 'A', 'B', 'B', 'C', 'C']  # 2 samples each
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(3)])
    classes = pd.Series(y, name='class')
    return df, classes


# ---------------------------------------------------------------------------
# train_classification_model — return signature
# ---------------------------------------------------------------------------


def test_train_returns_six_values(synthetic_data):
    """train_classification_model must return exactly 6 values."""
    train_df, train_classes = synthetic_data
    result = model_training.train_classification_model(
        train_df, train_classes, classifier_type='rf', perform_cv=False
    )
    assert len(result) == 6, f'Expected 6 return values, got {len(result)}'


def test_train_cv_results_dict_present(synthetic_data):
    train_df, train_classes = synthetic_data
    *_, cv_results = model_training.train_classification_model(
        train_df, train_classes, classifier_type='rf', perform_cv=False
    )
    assert isinstance(cv_results, dict)
    for key in (
        'cv_auroc_mean',
        'cv_auroc_std',
        'best_params',
        'best_cv_auroc',
    ):
        assert key in cv_results, f'cv_results missing key: {key}'


# ---------------------------------------------------------------------------
# train_classification_model — all classifiers (no CV for speed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'ctype,expected_cls',
    [
        ('rf', RandomForestClassifier),
        ('xgb', XGBClassifier),
        ('svm', SVC),
        ('mlp', MLPClassifier),
        ('nb', GaussianNB),
    ],
)
def test_train_all_classifiers(synthetic_data, ctype, expected_cls):
    train_df, train_classes = synthetic_data
    (
        clf,
        scaler,
        enc,
        df_scaled,
        text,
        cv_res,
    ) = model_training.train_classification_model(
        train_df, train_classes, classifier_type=ctype, perform_cv=False
    )
    assert isinstance(clf, expected_cls)
    assert isinstance(scaler, MinMaxScaler)
    assert isinstance(enc, LabelEncoder)
    assert df_scaled.shape == train_df.shape
    assert isinstance(text, str)


def test_train_unsupported_classifier_raises(synthetic_data):
    train_df, train_classes = synthetic_data
    with pytest.raises(ValueError):
        model_training.train_classification_model(
            train_df,
            train_classes,
            classifier_type='unsupported',
            perform_cv=False,
        )


# ---------------------------------------------------------------------------
# train_classification_model — CV with AUROC
# ---------------------------------------------------------------------------


def test_cv_auroc_reported(synthetic_data):
    """When perform_cv=True, cv_auroc_mean must be populated."""
    train_df, train_classes = synthetic_data
    *_, cv_results = model_training.train_classification_model(
        train_df,
        train_classes,
        classifier_type='rf',
        perform_cv=True,
        n_splits=3,
        n_repeats=1,
        seed=42,
    )
    assert cv_results['cv_auroc_mean'] is not None
    assert 0.0 <= cv_results['cv_auroc_mean'] <= 1.0


def test_cv_output_contains_auroc_line(synthetic_data):
    """The text log must include the AUROC metric name."""
    train_df, train_classes = synthetic_data
    (
        *other,
        output_text,
        cv_results,
    ) = model_training.train_classification_model(
        train_df,
        train_classes,
        classifier_type='rf',
        perform_cv=True,
        n_splits=3,
        n_repeats=1,
        seed=42,
    )
    assert 'roc_auc' in output_text.lower() or 'auroc' in output_text.lower()


def test_cv_hyperparameter_tuning_runs(synthetic_data):
    """best_params must be populated (non-empty) after tuning for rf/xgb/svm/mlp."""
    train_df, train_classes = synthetic_data
    *_, cv_results = model_training.train_classification_model(
        train_df,
        train_classes,
        classifier_type='rf',
        perform_cv=True,
        n_splits=3,
        n_repeats=1,
        seed=7,
    )
    # For rf, best_params should contain at least one key
    assert isinstance(cv_results['best_params'], dict)
    assert len(cv_results['best_params']) > 0


def test_cv_full_branch_scores(synthetic_data):
    train_df, train_classes = synthetic_data
    (
        clf,
        scaler,
        enc,
        df_scaled,
        output_text,
        cv_results,
    ) = model_training.train_classification_model(
        train_df,
        train_classes,
        classifier_type='rf',
        perform_cv=True,
        n_splits=3,
        n_repeats=2,
        seed=42,
    )
    assert 'Cross-validation scores:' in output_text
    assert 'accuracy:' in output_text
    assert 'f1_macro:' in output_text
    assert isinstance(clf, RandomForestClassifier)


# ---------------------------------------------------------------------------
# CV edge cases
# ---------------------------------------------------------------------------


def test_cv_single_sample_class_skipped(small_imbalanced_data):
    train_df, train_classes = small_imbalanced_data
    clf, _, _, _, output_text, _ = model_training.train_classification_model(
        train_df,
        train_classes,
        classifier_type='rf',
        perform_cv=True,
        n_splits=5,
        n_repeats=2,
    )
    assert 'Cannot perform k-fold cross-validation' in output_text
    assert 'Skipping cross-validation' in output_text
    assert isinstance(clf, RandomForestClassifier)


def test_cv_few_samples_reduces_splits(small_few_samples_data):
    train_df, train_classes = small_few_samples_data
    clf, _, _, _, output_text, _ = model_training.train_classification_model(
        train_df,
        train_classes,
        classifier_type='rf',
        perform_cv=True,
        n_splits=5,
        n_repeats=2,
    )
    assert 'Reducing n_splits from 5 to 2' in output_text
    assert isinstance(clf, RandomForestClassifier)


# ---------------------------------------------------------------------------
# predict_and_evaluate — return signature
# ---------------------------------------------------------------------------


def test_predict_returns_seven_values(synthetic_data):
    train_df, train_classes = synthetic_data
    clf, scaler, enc, _, _, _ = model_training.train_classification_model(
        train_df, train_classes, classifier_type='rf', perform_cv=False
    )
    name_class = [str(c) for c in enc.classes_]

    with tempfile.TemporaryDirectory() as tmp_dir:
        result = model_training.predict_and_evaluate(
            clf,
            scaler,
            enc,
            train_df,
            train_classes,
            name_class,
            tmp_dir,
        )
    assert len(result) == 7, f'Expected 7 return values, got {len(result)}'


def test_predict_auroc_populated(synthetic_data):
    train_df, train_classes = synthetic_data
    clf, scaler, enc, _, _, _ = model_training.train_classification_model(
        train_df, train_classes, classifier_type='rf', perform_cv=False
    )
    name_class = [str(c) for c in enc.classes_]

    with tempfile.TemporaryDirectory() as tmp_dir:
        *_, test_auroc = model_training.predict_and_evaluate(
            clf,
            scaler,
            enc,
            train_df,
            train_classes,
            name_class,
            tmp_dir,
        )
    assert test_auroc is not None
    assert 0.0 <= test_auroc <= 1.0


def test_predict_run_id_in_output(synthetic_data):
    train_df, train_classes = synthetic_data
    clf, scaler, enc, _, _, _ = model_training.train_classification_model(
        train_df, train_classes, classifier_type='rf', perform_cv=False
    )
    name_class = [str(c) for c in enc.classes_]

    with tempfile.TemporaryDirectory() as tmp_dir:
        _, _, _, complete_output, *_ = model_training.predict_and_evaluate(
            clf,
            scaler,
            enc,
            train_df,
            train_classes,
            name_class,
            tmp_dir,
            run_id='run_test_abc123',
        )
    assert 'run_test_abc123' in complete_output


def test_predict_with_previous_output(synthetic_data):
    train_df, train_classes = synthetic_data
    clf, scaler, enc, _, _, _ = model_training.train_classification_model(
        train_df, train_classes, classifier_type='rf', perform_cv=False
    )
    name_class = [str(c) for c in enc.classes_]
    prev = 'PREVIOUS_TRAINING_OUTPUT'

    with tempfile.TemporaryDirectory() as tmp_dir:
        _, _, _, complete_output, *_ = model_training.predict_and_evaluate(
            clf,
            scaler,
            enc,
            train_df,
            train_classes,
            name_class,
            tmp_dir,
            previous_output=prev,
        )
    assert prev in complete_output


# ---------------------------------------------------------------------------
# save_model / load_model — split_def stored and loaded
# ---------------------------------------------------------------------------


def test_save_load_model_contains_split_def(synthetic_data):
    train_df, train_classes = synthetic_data
    clf, scaler, enc, _, _, _ = model_training.train_classification_model(
        train_df, train_classes, classifier_type='rf', perform_cv=False
    )
    name_class = [str(c) for c in enc.classes_]
    split_def = {
        'seed': 42,
        'test_size': 0.2,
        'run_id': 'run_42_abcdefgh',
        'train_sample_ids': ['s1', 's2'],
        'test_sample_ids': ['s3'],
    }

    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        model_path = tmp.name

    try:
        model_training.save_model(
            clf,
            scaler,
            enc,
            split_def=split_def,
            output_path=model_path,
            name_class=name_class,
        )
        loaded = model_training.load_model(model_path)

        assert 'split_def' in loaded
        assert loaded['split_def']['seed'] == 42
        assert loaded['split_def']['run_id'] == 'run_42_abcdefgh'
        assert loaded['split_def']['test_sample_ids'] == ['s3']
        # Legacy 'test_data' key should NOT be present
        assert 'test_data' not in loaded
    finally:
        if os.path.exists(model_path):
            os.unlink(model_path)


def test_save_load_model_contains_cv_results(synthetic_data):
    train_df, train_classes = synthetic_data
    clf, scaler, enc, _, _, _ = model_training.train_classification_model(
        train_df, train_classes, classifier_type='rf', perform_cv=False
    )
    split_def = {
        'seed': 7,
        'test_size': 0.2,
        'run_id': 'run_7_xxxxxxxx',
        'train_sample_ids': [],
        'test_sample_ids': [],
    }
    cv_results = {
        'cv_auroc_mean': 0.95,
        'cv_auroc_std': 0.02,
        'best_params': {'n_estimators': 300},
        'best_cv_auroc': 0.96,
    }

    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        model_path = tmp.name

    try:
        model_training.save_model(
            clf,
            scaler,
            enc,
            split_def=split_def,
            output_path=model_path,
            cv_results=cv_results,
        )
        loaded = model_training.load_model(model_path)
        assert loaded['cv_results']['cv_auroc_mean'] == 0.95
        assert loaded['cv_results']['best_params']['n_estimators'] == 300
    finally:
        if os.path.exists(model_path):
            os.unlink(model_path)

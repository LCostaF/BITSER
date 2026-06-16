import io
import pickle
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from bitser.file_utils import save_output_to_file

# ---------------------------------------------------------------------------
# Hyperparameter grids (tuned strictly within CV; no test data involved)
# ---------------------------------------------------------------------------

_PARAM_GRIDS = {
    'rf': {
        'n_estimators': [100, 300],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
    },
    'xgb': {
        'n_estimators': [100, 300],
        'max_depth': [3, 6],
        'learning_rate': [0.1, 0.3],
    },
    'svm': {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['rbf', 'linear'],
    },
    'mlp': {
        'hidden_layer_sizes': [(100,), (100, 50)],
        'alpha': [0.0001, 0.001],
    },
    'nb': {},  # GaussianNB has no meaningful grid; skip tuning
}


def _build_base_classifier(classifier_type: str, seed: int):
    """Return an unfitted classifier with fixed random_state but default hyperparams."""
    if classifier_type == 'rf':
        return RandomForestClassifier(n_estimators=100, random_state=seed)
    elif classifier_type == 'xgb':
        return xgb.XGBClassifier(
            objective='multi:softmax',
            random_state=seed,
            n_jobs=-1,
            eval_metric='mlogloss',
        )
    elif classifier_type == 'svm':
        return SVC(
            probability=True,
            random_state=seed,
            decision_function_shape='ovr',
            gamma='scale',
        )
    elif classifier_type == 'mlp':
        return MLPClassifier(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            random_state=seed,
            max_iter=1000,
            early_stopping=True,
        )
    elif classifier_type == 'nb':
        return GaussianNB()
    else:
        raise ValueError(
            f'Unsupported classifier type: {classifier_type}. '
            f'Supported types are: rf, xgb, svm, mlp, nb'
        )


def _tune_hyperparameters(
    base_clf,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    classifier_type: str,
    n_splits: int,
    seed: int,
) -> tuple:
    """
    Run GridSearchCV on *X_train* / *y_train* using AUROC as the selection
    criterion.  Returns ``(best_estimator, best_params, best_auroc_score)``.

    If no parameter grid is defined (e.g. GaussianNB), returns the base
    classifier unchanged with an empty params dict and ``None`` score.
    """
    grid = _PARAM_GRIDS.get(classifier_type, {})
    if not grid:
        base_clf.fit(X_train, y_train)
        return base_clf, {}, None

    cv_inner = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    scoring = 'roc_auc_ovr_weighted'

    gs = GridSearchCV(
        base_clf,
        grid,
        scoring=scoring,
        cv=cv_inner,
        n_jobs=-1,
        refit=True,
        error_score='raise',
    )
    gs.fit(X_train, y_train)

    return gs.best_estimator_, gs.best_params_, gs.best_score_


def train_classification_model(
    train_df: pd.DataFrame,
    train_classes: pd.Series,
    classifier_type: str = 'xgb',
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: int = 7,
    perform_cv: bool = True,
):
    """
    Train a classification model on *train_df* / *train_classes*.

    Protocol (when ``perform_cv=True``):
    1. Scale features with MinMaxScaler fitted on training data only.
    2. Run stratified k-fold CV (``n_splits`` folds, ``n_repeats`` repeats)
       to measure generalisation; AUROC is the primary reported metric.
    3. Run GridSearchCV *within* a fresh inner-CV fold to pick hyperparameters
       (no test data is ever involved).
    4. Re-train the final model with best hyperparameters on the FULL training
       subset.

    Parameters
    ----------
    train_df : pd.DataFrame
        Feature matrix for training samples only.
    train_classes : pd.Series
        Class labels for training samples only.
    classifier_type : str
        One of ``rf``, ``xgb``, ``svm``, ``mlp``, ``nb``.
    n_splits : int
        Number of stratified CV folds (default 5).
    n_repeats : int
        Number of repeated CV rounds for variance estimation (default 1).
    seed : int
        Controls scaler, CV shuffling, classifier RNG, and hyperparameter
        search — full reproducibility.
    perform_cv : bool
        Set to ``False`` to skip CV and tuning (useful for unit tests).

    Returns
    -------
    classifier : fitted estimator
    min_max_scaler : MinMaxScaler
    label_encoder : LabelEncoder
    train_df_scaled : pd.DataFrame
    output_text : str
        Human-readable log of CV scores and selected hyperparameters.
    cv_results : dict
        ``{'cv_auroc_mean': float, 'cv_auroc_std': float,
           'best_params': dict, 'best_cv_auroc': float|None}``
        Always present; values are ``None`` when ``perform_cv=False``.
    """
    f = io.StringIO()
    cv_results = {
        'cv_auroc_mean': None,
        'cv_auroc_std': None,
        'best_params': {},
        'best_cv_auroc': None,
    }

    with redirect_stdout(f):
        min_max_scaler = preprocessing.MinMaxScaler()
        label_encoder = preprocessing.LabelEncoder()

        train_df_scaled = pd.DataFrame(
            min_max_scaler.fit_transform(train_df), columns=train_df.columns
        )

        label_encoder.fit(train_classes)
        y_train = label_encoder.transform(train_classes)
        num_class = len(label_encoder.classes_)

        # ------------------------------------------------------------------
        # Cross-validation (outer loop — diagnostics only)
        # ------------------------------------------------------------------
        if perform_cv:
            print('Performing cross-validation...')

            class_counts = np.bincount(y_train)
            min_class_samples = int(min(class_counts[class_counts > 0]))

            print(f'Smallest class has {min_class_samples} samples')

            if min_class_samples < 2:
                print(
                    'Warning: Cannot perform k-fold cross-validation because smallest class has too few samples'
                )
                print(
                    'Skipping cross-validation and proceeding with model training'
                )
            else:
                actual_n_splits = n_splits
                if min_class_samples < n_splits:
                    actual_n_splits = min_class_samples
                    print(
                        f'Warning: Reducing n_splits from {n_splits} to {actual_n_splits} due to small class size'
                    )

                # Build a *fresh* base estimator for CV scoring (not the
                # final one — avoids any data leakage through refitting)
                cv_clf = _build_base_classifier(classifier_type, seed)

                # Repeated stratified k-fold outer CV
                from sklearn.model_selection import RepeatedStratifiedKFold

                cross_val = RepeatedStratifiedKFold(
                    n_splits=actual_n_splits,
                    n_repeats=n_repeats,
                    random_state=seed,
                )

                scoring_metrics = [
                    'accuracy',
                    'precision_macro',
                    'recall_macro',
                    'f1_macro',
                    'roc_auc_ovr_weighted',
                ]

                cv_scores = cross_validate(
                    cv_clf,
                    train_df_scaled,
                    y_train,
                    scoring=scoring_metrics,
                    cv=cross_val,
                    n_jobs=-1,
                    error_score='raise',
                )

                print('\nCross-validation scores:')
                for metric, scores in cv_scores.items():
                    if metric.startswith('test_'):
                        label = metric[5:]
                        print(
                            f'{label}: {np.mean(scores):.3f} ± {np.std(scores):.3f}'
                        )

                auroc_scores = cv_scores.get(
                    'test_roc_auc_ovr_weighted', np.array([])
                )
                cv_results['cv_auroc_mean'] = (
                    float(np.mean(auroc_scores)) if len(auroc_scores) else None
                )
                cv_results['cv_auroc_std'] = (
                    float(np.std(auroc_scores)) if len(auroc_scores) else None
                )

            # --------------------------------------------------------------
            # Hyperparameter tuning (inner GridSearchCV on training data)
            # AUROC is the selection criterion; test set is never seen here.
            # Tuning is skipped when CV itself was skipped (too few samples).
            # --------------------------------------------------------------
            if min_class_samples < 2:
                # Cannot run inner CV either — just build and fit base model.
                classifier = _build_base_classifier(classifier_type, seed)
                if classifier_type == 'xgb':
                    classifier.set_params(num_class=num_class)
                classifier.fit(train_df_scaled, y_train)
                # Jump past the rest of the perform_cv block
                output_text = f.getvalue()
                print(output_text)
                return (
                    classifier,
                    min_max_scaler,
                    label_encoder,
                    train_df_scaled,
                    output_text,
                    cv_results,
                )

            print('\nRunning hyperparameter tuning (inner CV)...')
            base_clf = _build_base_classifier(classifier_type, seed)
            # For XGBoost multi-class we need num_class in the constructor
            if classifier_type == 'xgb':
                base_clf.set_params(num_class=num_class)

            inner_splits = min(actual_n_splits, 5)
            best_clf, best_params, best_score = _tune_hyperparameters(
                base_clf,
                train_df_scaled,
                y_train,
                classifier_type,
                n_splits=inner_splits,
                seed=seed,
            )
            cv_results['best_params'] = best_params
            cv_results['best_cv_auroc'] = (
                float(best_score) if best_score is not None else None
            )

            if best_params:
                print(f'Best hyperparameters: {best_params}')
            if best_score is not None:
                print(f'Best inner-CV AUROC: {best_score:.4f}')

            # The best_clf returned by GridSearchCV is already refit on the
            # full training fold — but we want it refit on the *full* training
            # set (not just the last inner fold).  Refit explicitly.
            classifier = best_clf
            print('\nRefitting best model on full training subset...')
            classifier.fit(train_df_scaled, y_train)

        else:
            # No CV requested — build base classifier and fit directly.
            classifier = _build_base_classifier(classifier_type, seed)
            if classifier_type == 'xgb':
                classifier.set_params(num_class=num_class)
            classifier.fit(train_df_scaled, y_train)

    output_text = f.getvalue()
    print(output_text)

    return (
        classifier,
        min_max_scaler,
        label_encoder,
        train_df_scaled,
        output_text,
        cv_results,
    )


def predict_and_evaluate(
    classifier,
    min_max_scaler,
    label_encoder,
    test_df: pd.DataFrame,
    test_classes: pd.Series,
    name_class: list,
    output_dir: str,
    run_id: str = '',
    train_df=None,
    previous_output: str = '',
    classifier_type: str = 'xgb',
    validation_df=None,
    validation_classes=None,
):
    """
    Evaluate a trained classifier strictly on the held-out test subset.

    The test subset must be reconstructed via ``split_manager.reconstruct_split``
    before being passed here — this function performs no splitting itself,
    guaranteeing zero leakage.

    Parameters
    ----------
    classifier : fitted estimator
    min_max_scaler : MinMaxScaler fitted on training data only
    label_encoder : LabelEncoder fitted on training labels
    test_df : pd.DataFrame
        Features for held-out test samples.
    test_classes : pd.Series
        True labels for held-out test samples.
    name_class : list
        Ordered class names corresponding to encoder classes.
    output_dir : str
        Directory where the results text file is written.
    run_id : str
        Run identifier (stored in output for traceability).
    train_df : pd.DataFrame, optional
        Used only for feature-importance display (column names).
    previous_output : str
        Training log text to prepend in the combined output file.
    classifier_type : str
        Short name of the classifier (``rf``, ``xgb``, …).
    validation_df, validation_classes : optional
        If supplied, an additional validation-set evaluation is appended.

    Returns
    -------
    classifier, min_max_scaler, label_encoder : unchanged artefacts
    complete_output : str
        Full log (training + evaluation).
    predictions : np.ndarray
        Predicted class labels (inverse-transformed) for each test sample.
    y_test : np.ndarray
        Numeric encoded true labels for test samples.
    test_auroc : float or None
        AUROC on the test set (None for binary if probability unavailable).
    """
    f = io.StringIO()
    test_auroc = None

    with redirect_stdout(f):
        if run_id:
            print(f'\nRun ID: {run_id}')
        print('\nEvaluating on test data...')

        test_df_scaled = pd.DataFrame(
            min_max_scaler.transform(test_df), columns=test_df.columns
        )
        y_test = label_encoder.transform(test_classes)
        y_pred_test = classifier.predict(test_df_scaled)

        target_names = [str(c) for c in name_class]

        print('\nTest set results:')
        print(
            f'Classification report:\n'
            f'{classification_report(y_test, y_pred_test, target_names=target_names)}'
        )

        # AUROC
        try:
            if hasattr(classifier, 'predict_proba'):
                y_prob = classifier.predict_proba(test_df_scaled)
            elif hasattr(classifier, 'decision_function'):
                y_prob = classifier.decision_function(test_df_scaled)
            else:
                y_prob = None

            if y_prob is not None:
                multi_class = (
                    'ovr' if len(label_encoder.classes_) > 2 else 'raise'
                )
                test_auroc = roc_auc_score(
                    y_test,
                    y_prob,
                    multi_class=multi_class
                    if len(label_encoder.classes_) > 2
                    else None,
                    average='weighted'
                    if len(label_encoder.classes_) > 2
                    else None,
                )
                print(f'Test AUROC (weighted OvR): {test_auroc:.4f}')
        except Exception as auroc_err:
            print(f'AUROC computation skipped: {auroc_err}')

        # Confusion matrix
        cm_test = confusion_matrix(y_test, y_pred_test)
        print('\nConfusion Matrix (Test Data):')
        print(cm_test)

        print('\nPer-class accuracies (Test Data):')
        cm_test_normalized = confusion_matrix(
            y_test, y_pred_test, normalize='true'
        )
        accuracy_per_class_test = cm_test_normalized.diagonal()
        for idx, accuracy in enumerate(accuracy_per_class_test):
            print(f'Class {name_class[idx]} accuracy: {accuracy:.4f}')

        # Optional validation set
        if validation_df is not None and validation_classes is not None:
            print('\nEvaluating on validation data...')
            validation_df_scaled = pd.DataFrame(
                min_max_scaler.transform(validation_df),
                columns=validation_df.columns,
            )
            y_val = label_encoder.transform(validation_classes)
            y_pred_val = classifier.predict(validation_df_scaled)

            print('\nValidation set results:')
            print(
                f'Classification report:\n'
                f'{classification_report(y_val, y_pred_val, target_names=target_names)}'
            )

            cm_val = confusion_matrix(y_val, y_pred_val)
            print('\nConfusion Matrix (Validation Data):')
            print(cm_val)

            print('\nPer-class accuracies (Validation Data):')
            cm_val_normalized = confusion_matrix(
                y_val, y_pred_val, normalize='true'
            )
            accuracy_per_class_val = cm_val_normalized.diagonal()
            for idx, accuracy in enumerate(accuracy_per_class_val):
                print(f'Class {name_class[idx]} accuracy: {accuracy:.4f}')

        # Feature importance
        if hasattr(classifier, 'feature_importances_'):
            feature_importances = classifier.feature_importances_
            feature_names = (
                train_df.columns if train_df is not None else test_df.columns
            )

            feature_importances_df = pd.DataFrame(
                {'Feature': feature_names, 'Importance': feature_importances}
            ).sort_values(by='Importance', ascending=False)

            derivatives = []
            for i in range(len(feature_importances_df) - 1):
                derivative = (
                    feature_importances_df['Importance'].iloc[i]
                    - feature_importances_df['Importance'].iloc[i + 1]
                )
                derivatives.append(derivative)
            derivatives.append(0.0)
            feature_importances_df['Derivative'] = derivatives

            pd.set_option('display.max_rows', None)
            print('\nOverall Feature Importance:')
            print(feature_importances_df)

    evaluation_output = f.getvalue()
    print(evaluation_output)

    complete_output = previous_output + evaluation_output
    save_output_to_file(complete_output, classifier_type, output_dir)

    predictions = label_encoder.inverse_transform(y_pred_test)

    return (
        classifier,
        min_max_scaler,
        label_encoder,
        complete_output,
        predictions,
        y_test,
        test_auroc,
    )


def save_model(
    classifier,
    min_max_scaler,
    label_encoder,
    split_def: dict,
    output_path: str = 'model.pkl',
    name_class=None,
    output_text: str = '',
    train_df_columns=None,
    cv_results: dict | None = None,
):
    """
    Persist the trained model together with run metadata.

    The ``split_def`` dictionary (from ``split_manager.split_metadata``)
    is stored inside the model file so that ``bitser predict`` can
    reconstruct the exact same held-out test set.

    Parameters
    ----------
    split_def : dict
        Must contain at least ``seed``, ``run_id``, ``train_sample_ids``,
        ``test_sample_ids`` produced by ``split_manager.split_metadata``.
    cv_results : dict, optional
        CV metrics and best hyperparameters from ``train_classification_model``.
    """
    model_data = {
        'classifier': classifier,
        'scaler': min_max_scaler,
        'encoder': label_encoder,
        'split_def': split_def,  # replaces old 'test_data'
        'name_class': name_class,
        'output_text': output_text,
        'train_df_columns': train_df_columns,
        'cv_results': cv_results or {},
    }
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)


def load_model(input_path: str = 'model.pkl') -> dict:
    """Load a model artefact produced by ``save_model``."""
    with open(input_path, 'rb') as f:
        return pickle.load(f)

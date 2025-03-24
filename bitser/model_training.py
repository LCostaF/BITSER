import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import io
from contextlib import redirect_stdout

from bitser.file_utils import save_output_to_file, save_validation_data


def train_classification_model(train_df, train_classes, classifier_type='rf',
                               n_splits=10, n_repeats=10, seed=7, perform_cv=True):
    """
    Train a classification model on the provided data.

    Parameters:
    -----------
    train_df : pandas.DataFrame
        DataFrame containing the features for the training dataset
    train_classes : pandas.Series
        Series containing the class labels for the training dataset
    classifier_type : str, default='rf'
        Flag for classifier type ('rf' for RandomForest, 'xgb' for XGBoost)
    n_splits : int, default=10
        Number of splits for k-fold cross-validation
    n_repeats : int, default=10
        Number of times to repeat the cross-validation process
    seed : int, default=7
        Random seed for reproducibility
    perform_cv : bool, default=True
        Whether to perform cross-validation during training

    Returns:
    --------
    classifier : trained classifier model
    min_max_scaler : Scaler used to normalize the features
    label_encoder : Encoder used to transform class labels into numerical values
    train_df_scaled : pandas.DataFrame
        Scaled training data (only needed for internal use)
    output_text : str
        Captured output from the training process
    """
    f = io.StringIO()
    with redirect_stdout(f):
        min_max_scaler = preprocessing.MinMaxScaler()
        label_encoder = preprocessing.LabelEncoder()

        train_df_scaled = pd.DataFrame(
            min_max_scaler.fit_transform(train_df),
            columns=train_df.columns
        )

        label_encoder.fit(train_classes)
        y_train = label_encoder.transform(train_classes)

        # Initialize classifier
        if classifier_type == 'rf':
            classifier = RandomForestClassifier(n_estimators=100, random_state=seed)
        elif classifier_type == 'xgb':
            num_class = len(label_encoder.classes_)
            classifier = xgb.XGBClassifier(objective='multi:softprob', random_state=seed, num_class=num_class,
                                           n_jobs=-1)
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")

        # Perform cross-validation if requested
        if perform_cv:
            print("Performing cross-validation...")
            cross_val = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
            cv_scores = cross_validate(
                classifier,
                train_df_scaled,
                y_train,
                scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
                cv=cross_val,
                n_jobs=-1,
                error_score="raise"
            )

            # Print cross-validation results
            print('\nCross-validation scores:')
            for metric, scores in cv_scores.items():
                if metric.startswith('test_'):
                    print(f'{metric[5:]}: {np.mean(scores):.3f} ± {np.std(scores):.3f}')

        # Train the final model on the full training set
        classifier.fit(train_df_scaled, y_train)

    output_text = f.getvalue()

    print(output_text)

    return classifier, min_max_scaler, label_encoder, train_df_scaled, output_text


def predict_and_evaluate(classifier, min_max_scaler, label_encoder, test_df, test_classes, name_class, train_df=None,
                         previous_output="", classifier_type='rf', validation_df=None, validation_classes=None, save_files=True):
    """
    Predict and evaluate a trained classifier on test and validation data.

    Parameters:
    -----------
    classifier : trained classifier model
    min_max_scaler : Scaler used to normalize the features
    label_encoder : Encoder used to transform class labels into numerical values
    test_df : pandas.DataFrame
        DataFrame containing the features for the test dataset
    test_classes : pandas.Series
        Series containing the class labels for the test dataset
    name_class : list
        List of unique class names corresponding to the class labels
    train_df : pandas.DataFrame, optional
        Original training DataFrame (used only for feature names in importance display)
    previous_output : str, default=""
        Previous output text to append to
    classifier_type : str, default='rf'
        Flag for classifier type
    validation_df : pandas.DataFrame, optional
        DataFrame containing the features for the validation dataset
    validation_classes : pandas.Series, optional
        Series containing the class labels for the validation dataset
    save_files : bool, default=True
        Whether to save output files

    Returns:
    --------
    classifier : trained classifier model
    min_max_scaler : Scaler used to normalize the features
    label_encoder : Encoder used to transform class labels into numerical values
    """
    f = io.StringIO()
    with redirect_stdout(f):
        print("\nEvaluating on test data...")
        test_df_scaled = pd.DataFrame(
            min_max_scaler.transform(test_df),
            columns=test_df.columns
        )
        y_test = label_encoder.transform(test_classes)
        y_pred_test = classifier.predict(test_df_scaled)

        print('\nTest set results:')
        print(f'Classification report:\n{classification_report(y_test, y_pred_test, target_names=name_class)}')

        print('\nPer-class accuracies (Test Data):')
        cm_test = confusion_matrix(y_test, y_pred_test, normalize="true")
        accuracy_per_class_test = cm_test.diagonal()
        for idx, accuracy in enumerate(accuracy_per_class_test):
            print(f'Class {name_class[idx]} accuracy: {accuracy:.4f}')

        if validation_df is not None and validation_classes is not None:
            print("\nEvaluating on validation data...")
            validation_df_scaled = pd.DataFrame(
                min_max_scaler.transform(validation_df),
                columns=validation_df.columns
            )
            y_val = label_encoder.transform(validation_classes)
            y_pred_val = classifier.predict(validation_df_scaled)

            print('\nValidation set results:')
            print(f'Classification report:\n{classification_report(y_val, y_pred_val, target_names=name_class)}')

            print('\nPer-class accuracies (Validation Data):')
            cm_val = confusion_matrix(y_val, y_pred_val, normalize="true")
            accuracy_per_class_val = cm_val.diagonal()
            for idx, accuracy in enumerate(accuracy_per_class_val):
                print(f'Class {name_class[idx]} accuracy: {accuracy:.4f}')

        # Print overall feature importance
        feature_importances = classifier.feature_importances_
        feature_names = train_df.columns if train_df is not None else test_df.columns
        feature_importances_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        pd.set_option('display.max_rows', None)
        print('\nOverall Feature Importance:')
        print(feature_importances_df)

    evaluation_output = f.getvalue()

    print(evaluation_output)

    complete_output = previous_output + evaluation_output

    if save_files:
        save_output_to_file(complete_output, classifier_type)

    return classifier, min_max_scaler, label_encoder


def run_classification(train_df, test_df, train_classes, test_classes, name_class, classifier_type='rf',
                       n_splits=10, n_repeats=10, seed=7, save_files=True):
    """
    Perform either cross-validation or train-test evaluation on a dataset.

    This function maintains the same interface as the original run_classification
    but delegates to the new split functions internally.

    If test_df is None, the function performs repeated stratified k-fold cross-validation
    after splitting the provided dataset into training, validation, and testing sets.
    If test_df is provided, the function performs a standard train-test evaluation.

    Parameters:
    -----------
    train_df : pandas.DataFrame
        DataFrame containing the features for the training dataset
    test_df : pandas.DataFrame, optional
        DataFrame containing the features for the test dataset. If None, cross-validation is performed.
    train_classes : pandas.Series
        Series containing the class labels for the training dataset
    test_classes : pandas.Series, optional
        Series containing the class labels for the test dataset
    name_class : list
        List of unique class names corresponding to the class labels
    classifier_type : str, default='rf'
        Flag for classifier type ('rf' for RandomForest, 'xgb' for XGBoost)
    n_splits : int, default=10
        Number of splits for k-fold cross-validation
    n_repeats : int, default=10
        Number of times to repeat the cross-validation process
    seed : int, default=7
        Random seed for reproducibility
    save_files : bool, default=True
        Whether to save output files

    Returns:
    --------
    classifier : trained classifier model
    min_max_scaler : Scaler used to normalize the features
    label_encoder : Encoder used to transform class labels into numerical values
    """
    f = io.StringIO()
    with redirect_stdout(f):
        if test_df is None:
            print("Performing cross-validation evaluation...")

            # Split data into train, validation, and test sets
            train_subset, test_subset, train_classes_subset, test_classes_subset = train_test_split(
                train_df, train_classes, test_size=0.2, random_state=seed, stratify=train_classes
            )
            train_subset, val_subset, train_classes_subset, val_classes_subset = train_test_split(
                train_subset, train_classes_subset, test_size=0.25, random_state=seed, stratify=train_classes_subset
            )

            test_df = test_subset
            test_classes = test_classes_subset
            validation_df = val_subset
            validation_classes = val_classes_subset

            if save_files:
                save_validation_data(validation_df, validation_classes, classifier_type)

        else:
            print("Performing train-test evaluation...")

            # Split training data into train and validation sets
            train_subset, val_subset, train_classes_subset, val_classes_subset = train_test_split(
                train_df, train_classes, test_size=0.2, random_state=seed, stratify=train_classes
            )

            validation_df = val_subset
            validation_classes = val_classes_subset

            if save_files:
                save_validation_data(validation_df, validation_classes, classifier_type)

    initial_output = f.getvalue()

    print(initial_output)

    # Cross-validation mode (test_df is None)
    if test_df is None:
        # Train model on training subset with cross-validation
        classifier, min_max_scaler, label_encoder, _, train_output = train_classification_model(
            train_subset, train_classes_subset, classifier_type, n_splits, n_repeats, seed, perform_cv=True
        )
    else:
        # Train model on full training data without cross-validation
        classifier, min_max_scaler, label_encoder, _, train_output = train_classification_model(
            train_subset, train_classes_subset, classifier_type, n_splits, n_repeats, seed, perform_cv=False
        )

    combined_output = initial_output + train_output

    return predict_and_evaluate(classifier, min_max_scaler, label_encoder, test_df, test_classes, name_class, train_df, combined_output, classifier_type, validation_df, validation_classes, save_files)

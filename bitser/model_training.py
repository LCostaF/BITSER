import io
import pickle
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from bitser.file_utils import save_output_to_file


def train_classification_model(
    train_df,
    train_classes,
    classifier_type='xgb',
    n_splits=10,
    n_repeats=10,
    seed=7,
    perform_cv=True,
):
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
            min_max_scaler.fit_transform(train_df), columns=train_df.columns
        )

        label_encoder.fit(train_classes)
        y_train = label_encoder.transform(train_classes)

        if classifier_type == 'rf':
            classifier = RandomForestClassifier(
                n_estimators=100, random_state=seed
            )
        elif classifier_type == 'xgb':
            num_class = len(label_encoder.classes_)
            classifier = xgb.XGBClassifier(
                objective='multi:softprob',
                random_state=seed,
                num_class=num_class,
                n_jobs=-1,
            )
        elif classifier_type == 'svm':
            classifier = SVC(
                probability=True,
                random_state=seed,
                decision_function_shape='ovr',
                gamma='scale',
            )
        elif classifier_type == 'mlp':
            classifier = MLPClassifier(
                hidden_layer_sizes=(100,),
                activation='relu',
                solver='adam',
                random_state=seed,
                max_iter=1000,
                early_stopping=True,
            )
        elif classifier_type == 'nb':
            classifier = GaussianNB()
        else:
            raise ValueError(
                f'Unsupported classifier type: {classifier_type}. '
                f'Supported types are: rf, xgb, svm, mlp, nb'
            )

        if perform_cv:
            print('Performing cross-validation...')

            class_counts = np.bincount(y_train)
            min_class_samples = min(class_counts[class_counts > 0])

            print(f'Smallest class has {min_class_samples} samples')

            if min_class_samples < 2:
                print(
                    'Warning: Cannot perform k-fold cross-validation because smallest class has too few samples'
                )
                print(
                    'Skipping cross-validation and proceeding with model training'
                )
            elif min_class_samples < n_splits:
                actual_n_splits = min(min_class_samples, n_splits)
                print(
                    f'Warning: Reducing n_splits from {n_splits} to {actual_n_splits} due to small class size'
                )

                print(
                    f'Warning: Reducing n_splits from {n_splits} to {actual_n_splits} due to small class size'
                )

                cross_val = RepeatedStratifiedKFold(
                    n_splits=actual_n_splits,
                    n_repeats=n_repeats,
                    random_state=seed,
                )

                cv_scores = cross_validate(
                    classifier,
                    train_df_scaled,
                    y_train,
                    scoring=[
                        'accuracy',
                        'precision_macro',
                        'recall_macro',
                        'f1_macro',
                    ],
                    cv=cross_val,
                    n_jobs=-1,
                    error_score='raise',
                )

                # Print cross-validation results
                print('\nCross-validation scores:')
                for metric, scores in cv_scores.items():
                    if metric.startswith('test_'):
                        print(
                            f'{metric[5:]}: {np.mean(scores):.3f} ± {np.std(scores):.3f}'
                        )

            else:
                cross_val = RepeatedStratifiedKFold(
                    n_splits=n_splits, n_repeats=n_repeats, random_state=seed
                )

                cv_scores = cross_validate(
                    classifier,
                    train_df_scaled,
                    y_train,
                    scoring=[
                        'accuracy',
                        'precision_macro',
                        'recall_macro',
                        'f1_macro',
                    ],
                    cv=cross_val,
                    n_jobs=-1,
                    error_score='raise',
                )

                print('\nCross-validation scores:')
                for metric, scores in cv_scores.items():
                    if metric.startswith('test_'):
                        print(
                            f'{metric[5:]}: {np.mean(scores):.3f} ± {np.std(scores):.3f}'
                        )

        classifier.fit(train_df_scaled, y_train)

    output_text = f.getvalue()

    print(output_text)

    return (
        classifier,
        min_max_scaler,
        label_encoder,
        train_df_scaled,
        output_text,
    )


def predict_and_evaluate(
    classifier,
    min_max_scaler,
    label_encoder,
    test_df,
    test_classes,
    name_class,
    train_df=None,
    previous_output='',
    classifier_type='xgb',
    validation_df=None,
    validation_classes=None,
    save_files=True,
):
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
        print('\nEvaluating on test data...')
        test_df_scaled = pd.DataFrame(
            min_max_scaler.transform(test_df), columns=test_df.columns
        )
        y_test = label_encoder.transform(test_classes)
        y_pred_test = classifier.predict(test_df_scaled)

        print('\nTest set results:')
        print(
            f'Classification report:\n{classification_report(y_test, y_pred_test, target_names=name_class)}'
        )

        # Print confusion matrix
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
                f'Classification report:\n{classification_report(y_val, y_pred_val, target_names=name_class)}'
            )

            # Print confusion matrix for validation data
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

        # Print overall feature importance
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

    if save_files:
        save_output_to_file(complete_output, classifier_type)

    return classifier, min_max_scaler, label_encoder, complete_output


def save_model(
    classifier,
    min_max_scaler,
    label_encoder,
    test_data=None,
    output_path='model.pkl',
    name_class=None,
    output_text='',
    train_df_columns=None,
):
    model_data = {
        'classifier': classifier,
        'scaler': min_max_scaler,
        'encoder': label_encoder,
        'test_data': test_data,
        'name_class': name_class,
        'output_text': output_text,
        'train_df_columns': train_df_columns,
    }
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)


def load_model(input_path='model.pkl'):
    with open(input_path, 'rb') as f:
        return pickle.load(f)

import os

from bitser.data_preprocessing import prepare_dataframe
from bitser.feature_extraction import extract_features_from_path
from bitser.model_training import run_classification


def main(
    train_dir,
    test_dir=None,
    translate_sequences=False,
    classifier_type='rf',
    flank=8,
    n_splits=10,
    n_repeats=10,
    seed=7,
):
    """
    # Main function to run the classification pipeline.
    :param train_dir: Path to directory containing the training data in FASTA format.
    :param test_dir: Path to directory containing the test data in FASTA format. If None, cross-validation is performed
    :param translate_sequences: Boolean to translate sequences or not
    :param classifier_type: Flag for classifier type
    :param flank: Size of the sliding window that runs through the sequence
    :param n_splits: Number of splits for k-fold cross-validation.
    :param n_repeats: Number of times to repeat the cross-validation process.
    :param seed: Random seed for reproducibility.
    :return: classifier: Trained RandomForestClassifier model
             scaler: Scaler used to normalize the features
             label_encoder: Encoder used to transform class labels into numerical values
    """
    train_features = extract_features_from_path(train_dir, flank, translate_sequences)
    train_df, train_classes, name_class = prepare_dataframe(train_features)

    if test_dir is None:
        print("Running cross-validation on training data...")
        return run_classification(train_df, None, train_classes, None, name_class, classifier_type)
    else:
        print("Running train-test evaluation...")
        test_features = extract_features_from_path(test_dir, flank, translate_sequences)
        test_df, test_classes, _ = prepare_dataframe(test_features)
        return run_classification(train_df, test_df, train_classes, test_classes, name_class, classifier_type)


if __name__ == "__main__":
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels
    TRAIN_DIR = os.path.join(project_dir, "dir_sequences", "seqs", "VOCs", "1k", "train")
    # TEST_DIR = os.path.join(project_dir, "dir_sequences", "seqs", "VOCs", "1k", "test_classes")
    TEST_DIR = None

    classifier, scaler, label_encoder = main(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        translate_sequences=False,
        flank=8,
        n_splits=10,
        n_repeats=10,
        seed=7
    )

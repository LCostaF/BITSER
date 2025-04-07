import os
from datetime import datetime


def get_project_dir():
    return os.path.dirname(os.path.abspath(__file__))


def create_results_directory():
    """
    Create the results directory if it doesn't exist.

    Returns:
    --------
    results_dir : str
        Path to the results directory
    """
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create results directory if it doesn't exist
    results_dir = os.path.join(project_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f'Created results directory: {results_dir}')

    return results_dir


def create_validation_directory():
    """
    Create the validation data directory if it doesn't exist.

    Returns:
    --------
    validation_dir : str
        Path to the validation data directory
    """
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create validation_data directory if it doesn't exist
    validation_dir = os.path.join(project_dir, 'validation_data')
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)
        print(f'Created validation data directory: {validation_dir}')

    return validation_dir


def save_output_to_file(output_text, classifier_type):
    """
    Save the captured output to a text file in the results directory.

    Parameters:
    -----------
    output_text : str
        The text to save to the file
    classifier_type : str
        Type of classifier used ('rf' or 'xgb')

    Returns:
    --------
    file_path : str
        Path to the saved file
    """
    results_dir = create_results_directory()

    # Generate filename with datetime and classifier type
    datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'{datetime_str}_results_{classifier_type}.txt'
    file_path = os.path.join(results_dir, file_name)

    # Write output to file
    with open(file_path, 'w') as f:
        f.write(output_text)

    print(f'Results saved to: {file_path}')

    return file_path


def save_validation_data(validation_df, validation_classes, classifier_type):
    """
    Save the validation data to a CSV file in the validation data directory.

    Parameters:
    -----------
    validation_df : pandas.DataFrame
        DataFrame containing the features for the validation dataset
    validation_classes : pandas.Series
        Series containing the class labels for the validation dataset
    classifier_type : str
        Type of classifier used ('rf' or 'xgb')
    """
    validation_dir = create_validation_directory()

    # Generate filename with datetime and classifier type
    datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'{datetime_str}_validation_data_{classifier_type}.csv'
    file_path = os.path.join(validation_dir, file_name)

    # Combine features and labels into a single DataFrame
    validation_data = validation_df.copy()
    validation_data['class'] = validation_classes

    # Save to CSV
    validation_data.to_csv(file_path, index=False)
    print(f'Validation data saved to: {file_path}')

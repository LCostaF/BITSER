import os
from datetime import datetime


def save_output_to_file(output_text, classifier_type, output_dir):
    if not output_dir:
        raise ValueError('output_dir must be provided')

    os.makedirs(output_dir, exist_ok=True)

    datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'{datetime_str}_results_{classifier_type}.txt'
    file_path = os.path.join(output_dir, file_name)

    with open(file_path, 'w') as f:
        f.write(output_text)

    return file_path


def save_validation_data(
    validation_df, validation_classes, classifier_type, output_dir
):
    if not output_dir:
        raise ValueError('output_dir must be provided')

    os.makedirs(output_dir, exist_ok=True)

    datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'{datetime_str}_validation_data_{classifier_type}.csv'
    file_path = os.path.join(output_dir, file_name)

    validation_data = validation_df.copy()
    validation_data['class'] = validation_classes

    validation_data.to_csv(file_path, index=False)


def save_prediction_report(report_df, classifier_type, output_dir):
    if not output_dir:
        raise ValueError('output_dir must be provided')

    os.makedirs(output_dir, exist_ok=True)

    datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'{datetime_str}_{classifier_type}_predictions.csv'
    file_path = os.path.join(output_dir, file_name)

    report_df.to_csv(file_path, index=False)
    return file_path

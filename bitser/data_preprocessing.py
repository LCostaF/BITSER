import numpy as np
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)


def prepare_dataframe(features):
    """
    # Prepare a DataFrame from the extracted features. function takes the extracted features (histogram, BWS, BWP, and
    # class labels) and converts them into a pandas DataFrame. It also preprocesses the data by handling infinite and
    # NaN values, and ensuring all data is numeric.

    :param features: 2D numpy array containing the extracted features. Columns include the histogram values, BWS, BWP,
    and the class label.
    :return: data_frame : pandas.DataFrame with the preprocessed features
             class_values : pandas.Series with the class labels for each sequence
             name_class : List of unique class names
    """
    col_name = ['HistLBP' + str(i + 1) for i in range(256)] + [
        'BWS',
        'BWP',
        'CLASS',
    ]

    # Create and process dataframe
    data_frame = pd.DataFrame(features, columns=col_name)

    # Extract class values before preprocessing
    class_values = data_frame['CLASS']
    name_class = np.unique(class_values).tolist()

    # Preprocess features
    data_frame.drop(columns=['CLASS'], axis=1, inplace=True)
    data_frame.replace([np.inf, -np.inf], 0, inplace=True)
    data_frame.replace(np.nan, 0, inplace=True)
    data_frame = data_frame.apply(pd.to_numeric, errors='coerce')
    data_frame.fillna(0, inplace=True)

    return data_frame, class_values, name_class

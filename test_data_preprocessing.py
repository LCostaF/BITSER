import numpy as np
import pandas as pd

from bitser import data_preprocessing


def test_prepare_dataframe():
    mock_features = np.array(
        [
            [1, 2, 3] + [0] * 253 + [0.75, 0.5, 'class_a'],
            [4, 5, 6] + [0] * 253 + [0.8, 0.6, 'class_b'],
            [7, np.inf, np.nan] + [0] * 253 + [0.9, -np.inf, 'class_a'],
        ],
        dtype=object,
    )

    df, class_values, name_class = data_preprocessing.prepare_dataframe(
        mock_features
    )

    assert isinstance(df, pd.DataFrame)
    assert isinstance(class_values, pd.Series)
    assert isinstance(name_class, list)

    assert df.shape == (3, 258)
    assert len(class_values) == 3
    assert len(name_class) == 2

    assert 'class_a' in name_class
    assert 'class_b' in name_class
    assert list(class_values) == ['class_a', 'class_b', 'class_a']

    # Check if there are no inf/nan
    assert np.isfinite(df.values).all()
    assert not np.isnan(df.values).any()

    # Check specific values
    assert df.iloc[0, 0] == 1
    assert df.iloc[1, 1] == 5
    assert df.iloc[0, -2] == 0.75
    assert df.iloc[1, -1] == 0.6

    # Check if inf/nan were replaced with 0
    assert df.iloc[2, 1] == 0
    assert df.iloc[2, 2] == 0
    assert df.iloc[2, -1] == 0


def test_prepare_dataframe_empty():
    mock_features = np.array([], dtype=object).reshape(0, 259)

    df, class_values, name_class = data_preprocessing.prepare_dataframe(
        mock_features
    )

    assert df.shape == (0, 258)
    assert len(class_values) == 0
    assert name_class == []


def test_column_names():
    mock_features = np.array([[0] * 256 + [0.5, 0.5, 'test']], dtype=object)

    df, _, _ = data_preprocessing.prepare_dataframe(mock_features)

    expected_columns = ['HistLBP' + str(i) for i in range(256)] + [
        'BWS',
        'BWP',
    ]
    assert list(df.columns) == expected_columns

    assert 'CLASS' not in df.columns

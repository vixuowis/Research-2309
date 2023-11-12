# function_import --------------------

import joblib
import pandas as pd
import numpy as np

# function_code --------------------

def estimate_mortgage(data):
    """
    Estimate the mortgage for a given housing using the housing's features.

    Args:
        data (pandas.DataFrame): The housing data with features.

    Returns:
        numpy.ndarray: The estimated mortgage.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    model = joblib.load('model.joblib')
    filtered_columns = config['features']
    data = data[filtered_columns]
    data.columns = [f'feat_{col}' for col in data.columns]
    predictions = model.predict(data)
    return predictions

# test_function_code --------------------

def test_estimate_mortgage():
    """
    Test the function estimate_mortgage.
    """
    # Test case 1: Normal case
    data = pd.DataFrame({'feature_1': [1, 2, 3],'feature_n': [4, 5, 6]})
    result = estimate_mortgage(data)
    assert isinstance(result, np.ndarray), 'The result should be a numpy array.'

    # Test case 2: Empty data
    data = pd.DataFrame()
    result = estimate_mortgage(data)
    assert isinstance(result, np.ndarray), 'The result should be a numpy array.'

    # Test case 3: Data with missing features
    data = pd.DataFrame({'feature_1': [1, 2, 3]})
    result = estimate_mortgage(data)
    assert isinstance(result, np.ndarray), 'The result should be a numpy array.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_estimate_mortgage()
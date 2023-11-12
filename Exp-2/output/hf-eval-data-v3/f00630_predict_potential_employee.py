# function_import --------------------

import joblib
import pandas as pd
import numpy as np

# function_code --------------------

def predict_potential_employee(file_path):
    """
    Predicts whether a candidate would be a potential employee based on a list of background information.

    Args:
        file_path (str): The path to the csv file containing the candidate's data.

    Returns:
        predictions (numpy.ndarray): The predicted labels for each candidate in the input data.

    Raises:
        FileNotFoundError: If the specified file or model does not exist.
    """
    model = joblib.load('model.joblib')
    data = pd.read_csv(file_path)
    selected_features = ['age', 'education', 'experience', 'skill1', 'skill2']
    data = data[selected_features]
    predictions = model.predict(data)
    return predictions

# test_function_code --------------------

def test_predict_potential_employee():
    """
    Tests the predict_potential_employee function.
    """
    # Test case: Valid file path
    try:
        predictions = predict_potential_employee('valid_file_path.csv')
        assert isinstance(predictions, np.ndarray)
    except FileNotFoundError:
        pass

    # Test case: Invalid file path
    try:
        predictions = predict_potential_employee('invalid_file_path.csv')
    except FileNotFoundError:
        assert True

    # Test case: File path is not a string
    try:
        predictions = predict_potential_employee(123)
    except TypeError:
        assert True

    return 'All Tests Passed'

# call_test_function_code --------------------

test_predict_potential_employee()
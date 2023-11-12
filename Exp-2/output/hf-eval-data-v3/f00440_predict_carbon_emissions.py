# function_import --------------------

import joblib
import pandas as pd
import numpy as np

# function_code --------------------

def predict_carbon_emissions(feat_x1, feat_x2, feat_x3):
    """
    Function to predict carbon emissions based on building features.

    Args:
        feat_x1 (float): The first feature of the building.
        feat_x2 (float): The second feature of the building.
        feat_x3 (float): The third feature of the building.

    Returns:
        predictions (array): The predicted carbon emissions categories for the buildings.

    Raises:
        FileNotFoundError: If the model file 'model.joblib' is not found.
    """
    model = joblib.load('model.joblib')
    input_data = pd.DataFrame({"feat_x1": [feat_x1],
                               "feat_x2": [feat_x2],
                               "feat_x3": [feat_x3]})
    predictions = model.predict(input_data)
    return predictions

# test_function_code --------------------

def test_predict_carbon_emissions():
    """
    Function to test the predict_carbon_emissions function.
    """
    # Test case 1
    predictions = predict_carbon_emissions(1.0, 2.0, 3.0)
    assert isinstance(predictions, np.ndarray), 'The prediction result should be a numpy array.'

    # Test case 2
    predictions = predict_carbon_emissions(4.0, 5.0, 6.0)
    assert isinstance(predictions, np.ndarray), 'The prediction result should be a numpy array.'

    # Test case 3
    predictions = predict_carbon_emissions(7.0, 8.0, 9.0)
    assert isinstance(predictions, np.ndarray), 'The prediction result should be a numpy array.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_predict_carbon_emissions()
# function_import --------------------

import json
import joblib
import pandas as pd
import numpy as np

# function_code --------------------

def estimate_carbon_emissions(data_file):
    """
    Estimate the carbon emissions of a specific device.

    Args:
        data_file (str): The path to the CSV data file with the device's idle power, standby power, and active power.

    Returns:
        numpy.ndarray: The estimated carbon emissions of the device.

    Raises:
        FileNotFoundError: If the model or data file does not exist.
    """
    model = joblib.load('model.joblib')
    config = json.load(open('config.json'))
    features = config['features']
    data = pd.read_csv(data_file)
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]
    predictions = model.predict(data)
    return predictions

# test_function_code --------------------

def test_estimate_carbon_emissions():
    """Tests the estimate_carbon_emissions function."""
    data_file = 'test_data.csv'
    predictions = estimate_carbon_emissions(data_file)
    assert isinstance(predictions, np.ndarray), 'The result should be a numpy array.'
    assert predictions.shape[0] > 0, 'The result array should not be empty.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_estimate_carbon_emissions()
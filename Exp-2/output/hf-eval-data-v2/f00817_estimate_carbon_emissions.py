# function_import --------------------

import json
import joblib
import pandas as pd

# function_code --------------------

def estimate_carbon_emissions(data_file):
    """
    Estimate the carbon emissions of a specific device.

    Args:
        data_file (str): The path to the CSV data file with the device's idle power, standby power, and active power.

    Returns:
        predictions (array): The estimated carbon emissions of the device.
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
    """
    Test the function estimate_carbon_emissions.
    """
    data_file = 'test_data.csv'
    predictions = estimate_carbon_emissions(data_file)
    assert predictions is not None, 'No predictions were made.'
    assert len(predictions) > 0, 'The predictions array is empty.'

# call_test_function_code --------------------

test_estimate_carbon_emissions()
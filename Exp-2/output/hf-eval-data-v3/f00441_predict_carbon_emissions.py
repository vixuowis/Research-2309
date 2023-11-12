# function_import --------------------

import joblib
import pandas as pd
import json
import numpy as np

# function_code --------------------

def predict_carbon_emissions(data_file):
    """
    Predicts the carbon emissions based on the tabular data of material consumption.

    Args:
        data_file (str): The path to the CSV file containing material consumption data.

    Returns:
        numpy.ndarray: The predicted carbon emissions.

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

def test_predict_carbon_emissions():
    """
    Tests the predict_carbon_emissions function.
    """
    # Test with a sample data file
    try:
        predictions = predict_carbon_emissions('sample_data.csv')
        assert isinstance(predictions, np.ndarray), 'The result is not a numpy array.'
        print('Test passed.')
    except FileNotFoundError:
        print('Test failed. The model or data file does not exist.')

# call_test_function_code --------------------

test_predict_carbon_emissions()
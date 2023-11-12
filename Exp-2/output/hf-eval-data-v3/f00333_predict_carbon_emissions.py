# function_import --------------------

import json
import joblib
import pandas as pd
import numpy as np

# function_code --------------------

def predict_carbon_emissions(data_file):
    """
    Load a pre-trained machine learning model and use it to predict carbon emissions.

    Args:
        data_file (str): The path to the CSV file containing the historical data.

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
    return model.predict(data)

# test_function_code --------------------

def test_predict_carbon_emissions():
    """
    Test the predict_carbon_emissions function.
    """
    # Test with a valid data file
    try:
        predictions = predict_carbon_emissions('test_data.csv')
        assert isinstance(predictions, np.ndarray)
    except FileNotFoundError:
        print('Test data file not found.')

    # Test with a non-existent data file
    try:
        predict_carbon_emissions('non_existent.csv')
    except FileNotFoundError:
        pass
    else:
        assert False, 'Expected a FileNotFoundError.'

    print('All tests passed.')

# call_test_function_code --------------------

test_predict_carbon_emissions()
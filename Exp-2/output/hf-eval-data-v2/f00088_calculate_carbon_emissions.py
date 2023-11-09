# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def calculate_carbon_emissions(data_file):
    """
    Calculate the carbon emissions for given data.
    
    Args:
        data_file (str): The path to the input data file in CSV format.
    
    Returns:
        numpy.ndarray: The predicted carbon emissions.
    
    Raises:
        FileNotFoundError: If the input data file or the model file does not exist.
        json.JSONDecodeError: If the configuration file is not a valid JSON.
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

def test_calculate_carbon_emissions():
    """
    Test the calculate_carbon_emissions function.
    """
    data_file = 'test_data.csv'
    predictions = calculate_carbon_emissions(data_file)
    
    assert isinstance(predictions, np.ndarray), 'The result should be a numpy array.'
    assert len(predictions) > 0, 'The result array should not be empty.'
    assert not np.isnan(predictions).any(), 'The result array should not contain NaN values.'

# call_test_function_code --------------------

test_calculate_carbon_emissions()
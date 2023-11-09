# function_import --------------------

import joblib
import json
import pandas as pd

# function_code --------------------

def predict_carbon_emissions(data_file):
    """
    This function predicts if a given set of input data will result in high carbon emissions or not.
    
    Args:
        data_file (str): The path to the csv file containing the input data.
    
    Returns:
        predictions (list): A list of predictions where '1' represents 'high carbon emissions' and '0' represents 'low carbon emissions'.
    
    Raises:
        FileNotFoundError: If the provided data file does not exist.
    """
    model = joblib.load('model.joblib')
    config = json.load(open('config.json'))
    features = config['features']
    data = pd.read_csv(data_file)
    data = data[features]
    predictions = model.predict(data)
    return predictions

# test_function_code --------------------

def test_predict_carbon_emissions():
    """
    This function tests the predict_carbon_emissions function by using a sample data file.
    """
    data_file = 'sample_data.csv'
    predictions = predict_carbon_emissions(data_file)
    assert isinstance(predictions, list), 'The output should be a list.'
    assert all(isinstance(i, (int, float)) for i in predictions), 'All elements in the output list should be integers or floats.'

# call_test_function_code --------------------

test_predict_carbon_emissions()
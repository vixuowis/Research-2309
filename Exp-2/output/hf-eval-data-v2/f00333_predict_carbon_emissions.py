# function_import --------------------

import json
import joblib
import pandas as pd

# function_code --------------------

def predict_carbon_emissions(data_file):
    """
    This function loads a pre-trained machine learning model and uses it to predict carbon emissions.
    
    Args:
        data_file (str): The path to the CSV file containing the historical data.
    
    Returns:
        predictions (array): An array of predicted carbon emissions.
    
    Raises:
        FileNotFoundError: If the specified data file does not exist.
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
    This function tests the predict_carbon_emissions function by comparing the predicted values with the actual values.
    
    Raises:
        AssertionError: If the predicted values are not close enough to the actual values.
    """
    predictions = predict_carbon_emissions('test_data.csv')
    actual_values = pd.read_csv('test_data.csv')['emissions']
    assert np.allclose(predictions, actual_values, rtol=1e-05, atol=1e-08)

# call_test_function_code --------------------

test_predict_carbon_emissions()
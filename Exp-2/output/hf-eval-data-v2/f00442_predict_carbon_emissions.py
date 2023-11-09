# function_import --------------------

import json
import joblib
import pandas as pd

# function_code --------------------

def predict_carbon_emissions(data_file):
    """
    This function predicts the carbon emissions of different facilities based on the provided data.
    
    Args:
        data_file (str): The path to the CSV file containing the data of the facilities.
    
    Returns:
        predictions (array): An array containing the predicted carbon emissions for each facility in the data set.
    
    Raises:
        FileNotFoundError: If the provided data file or the model file does not exist.
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
    This function tests the predict_carbon_emissions function by comparing the predicted carbon emissions with the actual carbon emissions.
    
    Raises:
        AssertionError: If the predicted carbon emissions are not close to the actual carbon emissions.
    """
    test_data = pd.read_csv('test_data.csv')
    actual_emissions = test_data['emissions']
    predicted_emissions = predict_carbon_emissions('test_data.csv')
    
    for actual, predicted in zip(actual_emissions, predicted_emissions):
        assert abs(actual - predicted) < 0.1, f'Expected {actual}, but got {predicted}'

# call_test_function_code --------------------

test_predict_carbon_emissions()
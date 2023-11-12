# function_import --------------------

import joblib
import json
import pandas as pd

# function_code --------------------

def predict_carbon_emissions(data_file):
    """
    Predicts carbon emissions based on the input data.

    Args:
        data_file (str): The path to the csv file containing the input data.

    Returns:
        predictions (list): A list of predictions where '1' represents high carbon emissions and '0' represents low carbon emissions.

    Raises:
        FileNotFoundError: If the model or config file does not exist.
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
    Tests the predict_carbon_emissions function.
    """
    # Test with a sample data file
    predictions = predict_carbon_emissions('sample_data.csv')
    assert isinstance(predictions, list), 'The result is not a list.'
    assert all(isinstance(i, (int, float)) for i in predictions), 'The predictions are not numbers.'
    # Test with a different data file
    predictions = predict_carbon_emissions('different_data.csv')
    assert isinstance(predictions, list), 'The result is not a list.'
    assert all(isinstance(i, (int, float)) for i in predictions), 'The predictions are not numbers.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_predict_carbon_emissions()
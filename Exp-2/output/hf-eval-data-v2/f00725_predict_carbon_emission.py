# function_import --------------------

import json
import joblib
import pandas as pd

# function_code --------------------

def predict_carbon_emission(data_file):
    """
    This function predicts whether a chemical plant is exceeding carbon emission limits.
    
    Args:
        data_file (str): The path to the CSV file containing the data collected from the plant.
    
    Returns:
        predictions (array): An array of predictions where 1 indicates exceeding carbon emission limits and 0 indicates otherwise.
    
    Raises:
        FileNotFoundError: If the provided data file or the model file does not exist.
        JSONDecodeError: If the configuration file is not a valid JSON.
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

def test_predict_carbon_emission():
    """
    This function tests the predict_carbon_emission function by using a sample data file.
    """
    predictions = predict_carbon_emission('sample_data.csv')
    assert predictions is not None, 'The function should return a prediction.'
    assert isinstance(predictions, np.ndarray), 'The function should return an array.'
    assert len(predictions) > 0, 'The function should return at least one prediction.'

# call_test_function_code --------------------

test_predict_carbon_emission()
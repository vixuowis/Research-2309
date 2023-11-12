# function_import --------------------

import joblib
import json
import pandas as pd
import numpy as np

# function_code --------------------

def predict_environmental_impact(data_file):
    """
    Predict the potential negative impact on the environment based on certain factors.

    Args:
        data_file (str): The path to the data file in csv format.

    Returns:
        predictions (numpy.ndarray): The predicted potential negative impact on the environment.

    Raises:
        FileNotFoundError: If the model or configuration file does not exist.
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

def test_predict_environmental_impact():
    """
    Test the function predict_environmental_impact.
    """
    data_file = 'test_data.csv'
    predictions = predict_environmental_impact(data_file)
    assert isinstance(predictions, np.ndarray), 'The result should be a numpy array.'
    assert predictions.shape[0] == 100, 'The number of predictions should be 100.'
    print('All Tests Passed')

# call_test_function_code --------------------

test_predict_environmental_impact()
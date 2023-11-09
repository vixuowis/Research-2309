# function_import --------------------

import joblib
import json
import pandas as pd

# function_code --------------------

def predict_environmental_impact(data_file):
    """
    Predict the potential negative impact on the environment based on certain factors.
    
    Args:
        data_file (str): The path to the csv file containing the data.
    
    Returns:
        predictions (array): The predicted environmental impact for each row in the input data.
    
    Raises:
        FileNotFoundError: If the specified data file does not exist.
        JSONDecodeError: If there is an error decoding the configuration file.
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
    Test the predict_environmental_impact function.
    """
    data_file = 'test_data.csv'
    predictions = predict_environmental_impact(data_file)
    assert predictions is not None, 'No predictions were made.'
    assert len(predictions) > 0, 'The number of predictions is zero.'

# call_test_function_code --------------------

test_predict_environmental_impact()
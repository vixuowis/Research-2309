# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def predict_carbon_emissions(data_file):
    """
    This function predicts the carbon emissions of a construction project based on the tabular data of material consumption.

    Args:
        data_file (str): The path to the CSV file containing the material consumption data.

    Returns:
        predictions (array): The predicted carbon emissions for each row in the input data.
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
    This function tests the predict_carbon_emissions function by using a sample dataset.
    """
    predictions = predict_carbon_emissions('sample_data.csv')
    assert predictions is not None, 'The function did not return any predictions.'
    assert len(predictions) > 0, 'The function did not return any predictions.'
    assert isinstance(predictions, np.ndarray), 'The function did not return the predictions as a numpy array.'

# call_test_function_code --------------------

test_predict_carbon_emissions()
# function_import --------------------

import json
import joblib
import pandas as pd

# function_code --------------------

def predict_carbon_emissions(data_file):
    """
    This function predicts carbon emissions for new data using a pre-trained model.

    Args:
        data_file (str): The path to the new data file in CSV format.

    Returns:
        numpy.ndarray: An array of carbon emission predictions for the corresponding inputs.
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
    This function tests the predict_carbon_emissions function by comparing the output with expected results.
    """
    predictions = predict_carbon_emissions('test_data.csv')
    expected_results = joblib.load('expected_results.joblib')
    assert np.allclose(predictions, expected_results, rtol=1e-05, atol=1e-08), 'Test failed.'

# call_test_function_code --------------------

test_predict_carbon_emissions()
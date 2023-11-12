# function_import --------------------

import json
import joblib
import pandas as pd
import numpy as np

# function_code --------------------

def predict_carbon_emissions(data_file):
    """
    This function predicts the carbon emissions based on the given dataset.

    Args:
        data_file (str): The path to the input data file in CSV format.

    Returns:
        numpy.ndarray: The predicted carbon emissions.

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

def test_predict_carbon_emissions():
    """
    This function tests the predict_carbon_emissions function.
    """
    # Test with a sample data file
    try:
        predictions = predict_carbon_emissions('sample_data.csv')
        assert isinstance(predictions, np.ndarray), 'The result is not a numpy array.'
        print('Test passed.')
    except FileNotFoundError:
        print('Test skipped due to missing file.')
    except Exception as e:
        print(f'Test failed due to: {e}')

# call_test_function_code --------------------

test_predict_carbon_emissions()
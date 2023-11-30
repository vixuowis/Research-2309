# function_import --------------------

import json
import joblib
import pandas as pd
import numpy as np

# function_code --------------------

def predict_carbon_emissions(data_file):
    """
    Load a pre-trained machine learning model and use it to predict carbon emissions.

    Args:
        data_file (str): The path to the CSV file containing the historical data.

    Returns:
        numpy.ndarray: The predicted carbon emissions.

    Raises:
        FileNotFoundError: If the model or data file does not exist.
    """
    if not os.path.exists(os.environ['MODEL_PATH']):
        raise FileNotFoundError('The model file does not exist')

    if not os.path.exists(data_file):
        raise FileNotFoundError('The data file does not exist')

    # load the model
    model = joblib.load(os.environ['MODEL_PATH'])

    # load the historical data
    X = pd.read_csv(data_file)

    # predict carbon emissions
    y_pred = model.predict(X)

    return y_pred


# test_function_code --------------------

def test_predict_carbon_emissions():
    """
    Test the predict_carbon_emissions function.
    """
    # Test with a valid data file
    try:
        predictions = predict_carbon_emissions('test_data.csv')
        assert isinstance(predictions, np.ndarray)
    except FileNotFoundError:
        print('Test data file not found.')

    # Test with a non-existent data file
    try:
        predict_carbon_emissions('non_existent.csv')
    except FileNotFoundError:
        pass
    else:
        assert False, 'Expected a FileNotFoundError.'

    print('All tests passed.')


# call_test_function_code --------------------

test_predict_carbon_emissions()
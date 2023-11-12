# function_import --------------------

import json
import joblib
import pandas as pd
import numpy as np

# function_code --------------------

def predict_carbon_emissions(data_file):
    """
    Predict the carbon emissions of several power plants based on their characteristics.

    Args:
        data_file (str): The path to the input data file. The file should be in CSV format.

    Returns:
        numpy.ndarray: The predicted carbon emissions for each power plant.

    Raises:
        FileNotFoundError: If the specified data file does not exist.
    """
    # Load the trained model
    model = joblib.load('model.joblib')

    # Load the configuration file
    config = json.load(open('config.json'))
    features = config['features']

    # Process the input data
    data = pd.read_csv(data_file)
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]

    # Make predictions and return the results
    return model.predict(data)

# test_function_code --------------------

def test_predict_carbon_emissions():
    """
    Test the predict_carbon_emissions function.
    """
    # Test with a valid data file
    predictions = predict_carbon_emissions('valid_data.csv')
    assert isinstance(predictions, np.ndarray), 'The result is not a numpy array.'

    # Test with a data file that does not exist
    try:
        predict_carbon_emissions('nonexistent_data.csv')
    except FileNotFoundError:
        pass
    else:
        assert False, 'The function did not raise a FileNotFoundError when the data file does not exist.'

    # Test with a data file that does not have the required features
    try:
        predict_carbon_emissions('invalid_data.csv')
    except KeyError:
        pass
    else:
        assert False, 'The function did not raise a KeyError when the data file does not have the required features.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_predict_carbon_emissions()
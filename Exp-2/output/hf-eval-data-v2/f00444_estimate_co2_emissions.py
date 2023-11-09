# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def estimate_co2_emissions(data_file: str, model_file: str = 'model.joblib', config_file: str = 'config.json') -> pd.DataFrame:
    """
    Estimate CO2 emissions based on historic data.

    Args:
        data_file (str): Path to the CSV file containing the historic data.
        model_file (str, optional): Path to the trained model file. Defaults to 'model.joblib'.
        config_file (str, optional): Path to the configuration file. Defaults to 'config.json'.

    Returns:
        pd.DataFrame: A data frame containing the estimated CO2 emissions.
    """
    model = joblib.load(model_file)
    config = json.load(open(config_file))
    features = config['features']
    data = pd.read_csv(data_file)
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]
    predictions = model.predict(data)
    return predictions

# test_function_code --------------------

def test_estimate_co2_emissions():
    """
    Test the function estimate_co2_emissions.
    """
    data_file = 'test_data.csv'
    model_file = 'model.joblib'
    config_file = 'config.json'
    predictions = estimate_co2_emissions(data_file, model_file, config_file)
    assert isinstance(predictions, pd.DataFrame), 'The result should be a DataFrame.'
    assert not predictions.empty, 'The result DataFrame should not be empty.'

# call_test_function_code --------------------

test_estimate_co2_emissions()
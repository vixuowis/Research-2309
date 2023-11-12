# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def estimate_co2_emissions(data_file: str, model_file: str, config_file: str) -> pd.DataFrame:
    """
    Estimate CO2 emissions based on historic data.

    Args:
        data_file (str): Path to the CSV file containing the historic data.
        model_file (str): Path to the trained model file.
        config_file (str): Path to the configuration file.

    Returns:
        pd.DataFrame: A DataFrame containing the estimated CO2 emissions.

    Raises:
        FileNotFoundError: If any of the input files are not found.
    """
    try:
        model = joblib.load(model_file)
        config = json.load(open(config_file))
        features = config['features']
        data = pd.read_csv(data_file)
        data = data[features]
        data.columns = ['feat_' + str(col) for col in data.columns]
        predictions = model.predict(data)
        return predictions
    except FileNotFoundError as e:
        print(f'File not found: {e}')

# test_function_code --------------------

def test_estimate_co2_emissions():
    """
    Test the estimate_co2_emissions function.
    """
    data_file = 'test_data.csv'
    model_file = 'test_model.joblib'
    config_file = 'test_config.json'
    try:
        predictions = estimate_co2_emissions(data_file, model_file, config_file)
        assert isinstance(predictions, pd.DataFrame), 'The result is not a DataFrame.'
        print('All Tests Passed')
    except FileNotFoundError as e:
        print(f'Test files not found: {e}')

# call_test_function_code --------------------

test_estimate_co2_emissions()
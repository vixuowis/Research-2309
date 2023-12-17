# requirements_file --------------------

import subprocess

requirements = ["joblib", "pandas"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import json
import joblib
import pandas as pd

# function_code --------------------

def estimate_co2_emissions(csv_file, model_file='model.joblib', config_file='config.json'):
    """
    Estimate CO2 emissions from the historic data provided in a CSV file using
    a trained regression model.

    Args:
        csv_file (str): The filepath to the CSV file containing the historic data.
        model_file (str): The filepath to the trained model file. Default is 'model.joblib'.
        config_file (str): The filepath to the configuration file. Default is 'config.json'.

    Returns:
        pandas.Series: A pandas Series containing the estimated CO2 emissions.

    Raises:
        FileNotFoundError: If any of the provided files are not found.
        KeyError: If required features are missing in the CSV file's data.
    """
    # Loading the trained model
    model = joblib.load(model_file)
    # Reading the model configuration
    config = json.load(open(config_file))
    features = config['features']
    # Reading the historic data
    data = pd.read_csv(csv_file)
    # Checking if all required features are present
    if not all(feature in data.columns for feature in features):
        raise KeyError('Missing required features in the data.')
    # Selecting and renaming the required features
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]
    # Predicting the CO2 emissions
    predictions = model.predict(data)
    return predictions

# test_function_code --------------------

def test_estimate_co2_emissions():
    print("Testing started.")
    try:
        # Test case 1: Correct CSV and model provided
        print("Testing case [1/3] started.")
        predictions = estimate_co2_emissions('client_data.csv')
        assert not predictions.empty, "Test case [1/3] failed: No predictions returned."

        # Test case 2: Non-existent CSV file
        print("Testing case [2/3] started.")
        try:
            predictions = estimate_co2_emissions('non_existent.csv')
            assert False, "Test case [2/3] failed: FileNotFoundError not raised."
        except FileNotFoundError:
            pass

        # Test case 3: CSV missing required columns
        print("Testing case [3/3] started.")
        try:
            predictions = estimate_co2_emissions('incomplete_data.csv')
            assert False, "Test case [3/3] failed: KeyError not raised."
        except KeyError:
            pass
    finally:
        print("Testing finished.")

# call_test_function_line --------------------

test_estimate_co2_emissions()
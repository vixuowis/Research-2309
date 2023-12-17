# requirements_file --------------------

import subprocess

requirements = ["joblib", "json", "pandas"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import joblib
import json
import pandas as pd

# function_code --------------------

def predict_carbon_emissions(config_path, data_path):
    """
    Predict if the input data will result in high or low carbon emissions.

    Args:
        config_path (str): The file path to the config.json containing model features.
        data_path (str): The file path to the data.csv containing the input data.

    Returns:
        list: A list of predictions with 'high carbon emissions' or 'low carbon emissions'.

    Raises:
        FileNotFoundError: If the config.json or data.csv file does not exist.
    """
    # Load the pre-trained model
    model = joblib.load('model.joblib')
    
    # Load the configuration for features
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    # Extract features and prepare the input data
    features = config['features']
    data = pd.read_csv(data_path)
    data = data[features]
    
    # Predict and return the results
    predictions = model.predict(data)
    return ['high carbon emissions' if pred else 'low carbon emissions' for pred in predictions]

# test_function_code --------------------

def test_predict_carbon_emissions():
    print("Testing started.")
    # Set up test data and expected results
    test_config_path = 'test_config.json'
    test_data_path = 'test_data.csv'
    expected_results = ['high carbon emissions', 'low carbon emissions']

    # Testing case 1: Correct paths
    print("Testing case [1/2] started.")
    predictions = predict_carbon_emissions(test_config_path, test_data_path)
    assert predictions == expected_results, f"Test case [1/2] failed: Expected {expected_results}, got {predictions}"

    # Testing case 2: File not found
    print("Testing case [2/2] started.")
    try:
        predict_carbon_emissions('invalid_config.json', 'invalid_data.csv')
        assert False, "Test case [2/2] failed: FileNotFoundError expected"
    except FileNotFoundError:
        pass
    print("Testing finished.")

# call_test_function_line --------------------

test_predict_carbon_emissions()
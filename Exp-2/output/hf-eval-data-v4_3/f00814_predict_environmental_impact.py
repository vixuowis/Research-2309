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

def predict_environmental_impact(data_csv, config_json):
    """
    Predict the potential negative environmental impact using a pre-trained model.

    Args:
        data_csv (str): The CSV file path containing the input data for prediction.
        config_json (str): The JSON file path containing the configuration for selecting features.

    Returns:
        list: Predictions of negative environmental impact.

    Raises:
        FileNotFoundError: If the specified files are not found.
        KeyError: If the required keys are not present in the config file.
    """
    model = joblib.load('model.joblib')
    with open(config_json, 'r') as file:
        config = json.load(file)
    features = config['features']

    data = pd.read_csv(data_csv)
    data = data[features]

    # Preprocessing: renaming the columns
    data.columns = ['feat_' + str(col) for col in data.columns]
    predictions = model.predict(data)
    return predictions.tolist()

# test_function_code --------------------

def test_predict_environmental_impact():
    print("Testing started.")

    # Test case 1: Test with sample data
    print("Testing case [1/2] started.")
    predictions = predict_environmental_impact('test_data.csv', 'test_config.json')
    assert isinstance(predictions, list), "Test case [1/2] failed: predictions should be a list."

    # Test case 2: Test with non-existent file paths
    print("Testing case [2/2] started.")
    try:
        predict_environmental_impact('non_existent_data.csv', 'non_existent_config.json')
        assert False, "Test case [2/2] failed: FileNotFoundError not raised."
    except FileNotFoundError:
        pass  # Expected exception
    print("Testing finished.")

# call_test_function_line --------------------

test_predict_environmental_impact()
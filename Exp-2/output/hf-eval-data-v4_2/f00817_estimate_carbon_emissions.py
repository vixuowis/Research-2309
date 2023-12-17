# requirements_file --------------------

!pip install -U joblib pandas

# function_import --------------------

import json
import joblib
import pandas as pd

# function_code --------------------

def estimate_carbon_emissions(model_path, config_path, data_path):
    """
    Estimate carbon emissions of a device using a pre-trained regression model.

    Args:
        model_path (str): The path to the pre-trained model file.
        config_path (str): The path to the configuration file with feature info.
        data_path (str): The path to the CSV data file containing device measurements.

    Returns:
        list: The estimated carbon emissions.

    Raises:
        FileNotFoundError: If any of the provided file paths do not exist.
        ValueError: If the input data is not properly formatted.
    """
    model = joblib.load(model_path)
    with open(config_path) as config_file:
        config = json.load(config_file)
    features = config['features']
    data = pd.read_csv(data_path)
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]
    return model.predict(data).tolist()

# test_function_code --------------------

def test_estimate_carbon_emissions():
    print("Testing started.")
    expected_result = [...fill_with_appropriate_values...]
    actual_result = estimate_carbon_emissions('model.joblib', 'config.json', 'data.csv')

    # Test Case: Check if the actual result matches the expected result
    print("Testing case [1/1] started.")
    assert actual_result == expected_result, f"Test case [1/1] failed: Expected {expected_result}, got {actual_result}"
    print("Testing finished.")

# call_test_function_line --------------------

test_estimate_carbon_emissions()
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

def predict_carbon_emissions(model_filepath, config_filepath, data_filepath):
    """Predict carbon emissions for new data using a pre-trained model.

    Args:
        model_filepath (str): The file path to the pre-trained joblib model.
        config_filepath (str): The file path to the config.json containing feature info.
        data_filepath (str): The file path to the new data (CSV) for prediction.

    Returns:
        list: An array of predicted carbon emissions.

    Raises:
        FileNotFoundError: If any of the provided file paths do not exist.
    """
    model = joblib.load(model_filepath)
    with open(config_filepath) as f:
        config = json.load(f)
    features = config['features']
    data = pd.read_csv(data_filepath)
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]
    predictions = model.predict(data)
    return predictions.tolist()

# test_function_code --------------------

def test_predict_carbon_emissions():
    print("Testing started.")
    # Test case 1: Correct model, config, and data paths
    print("Testing case [1/1] started.")
    assert type(predict_carbon_emissions('model.joblib', 'config.json', 'data.csv')) is list, f"Test case [1/1] failed: Function should return a list of predictions."
    print("Testing finished.")

# call_test_function_line --------------------

test_predict_carbon_emissions()
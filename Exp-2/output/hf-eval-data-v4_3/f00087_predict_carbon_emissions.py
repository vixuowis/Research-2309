# requirements_file --------------------

import subprocess

requirements = ["joblib", "pandas"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def predict_carbon_emissions(data_csv_path, model_joblib_path, config_json_path):
    """Load a pretrained tabular regression model and predict carbon emissions based on the input data.

    Args:
        data_csv_path (str): The file path to the CSV containing the input features.
        model_joblib_path (str): The file path to the pretrained model in joblib format.
        config_json_path (str): The file path to the JSON configuration for feature selection.

    Returns:
        list: A list of predicted carbon emissions.

    Raises:
        FileNotFoundError: If any of the provided file paths do not exist.
    """
    # Load the configuration for feature selection
    config = json.load(open(config_json_path))
    features = config['features']
    
    # Load the dataset from a CSV file
    data = pd.read_csv(data_csv_path)
    
    # Select the important features and preprocess the input data
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]
    
    # Load the pretrained model
    model = joblib.load(model_joblib_path)
    
    # Predict carbon emissions
    predictions = model.predict(data)
    return predictions.tolist()

# test_function_code --------------------

def test_predict_carbon_emissions():
    print("Testing started.")
    # Test case 1: Valid input CSV and model
    print("Testing case [1/1] started.")
    predicted = predict_carbon_emissions('data.csv', 'model.joblib', 'config.json')
    assert isinstance(predicted, list), "Test case [1/1] failed: Predicted output should be a list."
    print("Testing finished.")

# call_test_function_line --------------------

test_predict_carbon_emissions()
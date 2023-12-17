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

def predict_carbon_emissions(model_path, config_path, data_path):
    """
    Load a regression model and predicts carbon emissions for a new line of electric vehicles.

    Args:
        model_path (str): The file path for the pre-trained joblib model.
        config_path (str): The file path for the configuration file containing features.
        data_path (str): The file path for the data on new electric vehicles.

    Returns:
        pandas.Series: A Pandas Series object containing the predicted carbon emissions.

    Raises:
        ValueError: If the specified file paths are invalid or the data is not as expected.
        FileNotFoundError: If any of the files are not found.
    """
    # Load the pre-trained regression model
    model = joblib.load(model_path)
    
    # Load the configuration for features
    config = json.load(open(config_path))
    features = config['features']

    # Load the new vehicle data
    data = pd.read_csv(data_path)
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]

    # Predict carbon emissions
    predictions = model.predict(data)
    return pd.Series(predictions)

# test_function_code --------------------

def test_predict_carbon_emissions():
    print("Testing started.")
    model_path = 'model.joblib'
    config_path = 'config.json'
    data_path = 'new_vehicle_data.csv'

    # Load a sample data for testing
    sample_data = pd.read_csv(data_path).iloc[0:3]  # To avoid using full dataset
    sample_data.to_csv('sample_vehicle_data.csv', index=False)

    # Testing case 1: Valid process
    print("Testing case [1/1] started.")
    try:
        predictions = predict_carbon_emissions(model_path, config_path, 'sample_vehicle_data.csv')
        assert len(predictions) == 3, f"Test case [1/1] failed: Expected 3 predictions, got {len(predictions)}"
    except Exception as e:
        assert False, f"Test case [1/1] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_predict_carbon_emissions()
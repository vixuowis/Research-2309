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

def predict_titanic_survival(model_joblib, config_json, data_csv):
    """
    Predict the survival of passengers on the Titanic.

    Args:
        model_joblib (str): The path to the saved joblib model file.
        config_json (str): The path to the JSON config file with feature names.
        data_csv (str): The path to the CSV file containing the passenger data.

    Returns:
        pandas.Series: A series with the survival predictions for each passenger.

    Raises:
        FileNotFoundError: If any of the provided file paths do not exist.
    """
    # Load the Titanic survival prediction model
    model = joblib.load(model_joblib)

    # Load the configuration for feature names
    with open(config_json, 'r') as f:
        config = json.load(f)
        features = config['features']

    # Read the data and select the required features
    data = pd.read_csv(data_csv)
    data = data[features]
    # Rename columns as per model's config
    data.columns = ['feat_' + str(col) for col in data.columns]

    # Predict the survival probabilities
    predictions = model.predict(data)
    return predictions

# test_function_code --------------------

def test_predict_titanic_survival():
    print("Testing started.")
    # Assuming a sample CSV file and a model/config are available for testing
    sample_model = 'test_model.joblib'
    sample_config = 'test_config.json'
    sample_data = 'test_data.csv'

    print("Testing case [1/1] started.")
    try:
        predictions = predict_titanic_survival(sample_model, sample_config, sample_data)
        assert isinstance(predictions, pd.Series), "The function should return a pandas Series."
    except FileNotFoundError as e:
        assert False, f"Test case failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_predict_titanic_survival()
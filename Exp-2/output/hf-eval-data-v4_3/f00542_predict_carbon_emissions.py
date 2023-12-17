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

def predict_carbon_emissions(data_csv_path, model_path='model.joblib', config_path='config.json'):
    """
    Predict carbon emissions for the input dataset.

    Args:
        data_csv_path (str): Path to the CSV file containing the dataset.
        model_path (str): Path to the saved model file. Default is 'model.joblib'.
        config_path (str): Path to the configuration file. Default is 'config.json'.

    Returns:
        pandas.DataFrame: A DataFrame containing the predictions.

    Raises:
        FileNotFoundError: If any of the specified files are not found.
        KeyError: If the expected features are not found in the dataset.
    """
    # Load the regression model
    model = joblib.load(model_path)

    # Load the configuration file with features
    with open(config_path) as config_file:
        config = json.load(config_file)
    features = config['features']

    # Read the input dataset
    data = pd.read_csv(data_csv_path)
    if not all(feature in data.columns for feature in features):
        raise KeyError('Dataset does not contain all required features.')

    # Preprocess the data
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]

    # Make predictions
    predictions = model.predict(data)
    return pd.DataFrame(predictions, columns=['Predicted Carbon Emissions'])

# test_function_code --------------------

def test_predict_carbon_emissions():
    print("Testing started.")
    sample_data_path = 'sample_data.csv'  # Path to a sample CSV file for testing
    model_path = 'model.joblib'  # Path to the trained model file
    config_path = 'config.json'  # Path to the config file containing the features

    # Expected features based on the configuration
    with open(config_path) as config_file:
        config = json.load(config_file)
    expected_features = config['features']

    # Load sample test data
    test_data = pd.read_csv(sample_data_path)

    # Test case 1: Correct prediction flow
    print("Testing case [1/3] started.")
    try:
        predictions = predict_carbon_emissions(sample_data_path, model_path, config_path)
        assert len(predictions) == len(test_data), f"Test case [1/3] failed: Prediction length mismatch."
        print("Test case [1/3] passed.")
    except Exception as e:
        print(f"Test case [1/3] failed: {e}")

    # Test case 2: Missing features in the dataset
    print("Testing case [2/3] started.")
    try:
        # Remove one expected feature to simulate missing data
        modified_data = test_data.drop(columns=[expected_features[0]])
        modified_data.to_csv(sample_data_path, index=False)
        predict_carbon_emissions(sample_data_path, model_path, config_path)
        print(f"Test case [2/3] failed: No KeyError was raised for missing features.")
    except KeyError:
        print("Test case [2/3] passed for missing features.")

    # Test case 3: File not found
    print("Testing case [3/3] started.")
    try:
        predict_carbon_emissions('non_existent_data.csv', model_path, config_path)
        print(f"Test case [3/3] failed: No FileNotFoundError was raised.")
    except FileNotFoundError:
        print("Test case [3/3] passed for file not found.")
    print("Testing finished.")

# call_test_function_line --------------------

test_predict_carbon_emissions()
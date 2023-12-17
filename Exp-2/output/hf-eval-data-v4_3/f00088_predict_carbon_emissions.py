# requirements_file --------------------

import subprocess

requirements = ["joblib", "pandas", "numpy"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def predict_carbon_emissions(data_csv_path, config_json_path, model_joblib_path):
    """
    Predicts carbon emissions based on the input CSV data.

    Args:
        data_csv_path (str): The file path to the CSV file containing the production data.
        config_json_path (str): The file path to the JSON configuration file specifying the model features.
        model_joblib_path (str): The file path to the Joblib file of the pre-trained model.

    Returns:
        numpy.ndarray: An array containing the predicted carbon emissions for the input data.

    Raises:
        FileNotFoundError: If any of the files at the specified paths do not exist.
        ValueError: If the input data does not contain the required features.
    """
    # Load the pre-trained model using joblib
    model = joblib.load(model_joblib_path)

    # Load the configuration file containing the selected features for the model
    with open(config_json_path) as config_file:
        config = json.load(config_file)
    features = config['features']

    # Load the input data (CSV format) using pandas
    data = pd.read_csv(data_csv_path)
    if any(feat not in data.columns for feat in features):
        raise ValueError("Input data CSV is missing one or more required features.")

    # Extract only the required features from the config file
    data = data[features]

    # Apply column naming convention by prefixing each feature column name with "feat_"
    data.columns = ['feat_' + str(col) for col in data.columns]

    # Use the loaded model to predict carbon emissions for the given data
    predictions = model.predict(data)

    return predictions

# test_function_code --------------------

def test_predict_carbon_emissions():
    print("Testing started.")
    
    # Test Case 1: Correct data and configuration
    print("Testing case [1/3] started.")
    predictions = predict_carbon_emissions('sample_data.csv', 'sample_config.json', 'sample_model.joblib')
    assert isinstance(predictions, np.ndarray), f"Test case [1/3] failed: Predictions type is not numpy.ndarray"
    
    # Test Case 2: Non-existing CSV file
    print("Testing case [2/3] started.")
    try:
        predict_carbon_emissions('non_existing.csv', 'sample_config.json', 'sample_model.joblib')
        assert False, "Test case [2/3] failed: FileNotFoundError not raised for missing CSV file."
    except FileNotFoundError:
        assert True
    
    # Test Case 3: Data missing required features
    print("Testing case [3/3] started.")
    try:
        predict_carbon_emissions('incomplete_data.csv', 'sample_config.json', 'sample_model.joblib')
        assert False, "Test case [3/3] failed: ValueError not raised for data missing features."
    except ValueError:
        assert True
    
    print("Testing finished.")

# Running the Test function
test_predict_carbon_emissions()

# call_test_function_line --------------------

test_predict_carbon_emissions()
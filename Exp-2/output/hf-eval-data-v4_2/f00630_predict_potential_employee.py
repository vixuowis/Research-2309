# requirements_file --------------------

!pip install -U joblib pandas

# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def predict_potential_employee(config_path, data_path):
    """
    Predict if candidates from the dataset are potential employees.

    Args:
        config_path (str): The file path to the configuration JSON containing model features.
        data_path (str): The file path to the candidate data CSV.

    Returns:
        List[bool]: A list of predictions where True indicates a potential employee.

    Raises:
        FileNotFoundError: If the configuration file or data CSV does not exist.
        KeyError: If the required features are not found in the data.
    """
    # Load the model and configuration
    model = joblib.load('model.joblib')
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    features = config['features']
    
    # Load and prepare the data
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError('Data file not found at specified path.')
    
    if not all(feature in data.columns for feature in features):
        raise KeyError('One or more required features are missing from the data.')
    
    # Select the relevant features and predict
    data = data[features]
    predictions = model.predict(data)
    
    return list(predictions == 1)

# test_function_code --------------------

def test_predict_potential_employee():
    print("Testing started.")
    # Config file for the model, contains features needed for prediction
    config_path = 'test_config.json'
    # Prepare a small dataset for testing
    data_path = 'test_data.csv'

    # Testing case [1/1] started
    print("Testing case [1/1] started.")
    predicted = predict_potential_employee(config_path, data_path)
    expected = [True, False, True]  # Expected results based on the test dataset
    assert predicted == expected, f"Test case [1/1] failed: expected {expected}, got {predicted}"
    print("Testing finished.")

# call_test_function_line --------------------

test_predict_potential_employee()
# requirements_file --------------------

import subprocess

requirements = ["json", "joblib", "pandas"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------



# function_code --------------------

def check_emissions(model_path, config_path, data_path):
    """
    Checks whether a chemical plant is exceeding carbon emission limits based on a CSV file containing data collected.

    Args:
        model_path (str): The path to the stored model joblib file.
        config_path (str): The path to the configuration json file.
        data_path (str): The path to the CSV data file.

    Returns:
        list[bool]: A list of boolean values indicating whether each row in the data is exceeding the limits. True for exceeding, False otherwise.

    Raises:
        FileNotFoundError: If any of the given file paths do not exist.
        Exception: If model prediction fails.
    """
    import json
    import joblib
    import pandas as pd

    # Load the classifier model
    model = joblib.load(model_path)

    # Load the configuration file
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    # Extract feature names from the configuration
    features = config['features']

    # Load and process the data
    data = pd.read_csv(data_path)
    if not all(feature in data.columns for feature in features):
        raise ValueError('Data file is missing required features.')
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]

    try:
        # Predict using the loaded model
        predictions = model.predict(data)
        return [bool(pred) for pred in predictions]
    except Exception as e:
        raise Exception(f'Model prediction failed: {e}')

# test_function_code --------------------

def test_check_emissions():
    print("Testing started.")

    # Test case 1: All files exist and predictions are successful
    print("Testing case [1/3] started.")
    try:
        predictions = check_emissions('test_model.joblib', 'test_config.json', 'test_data.csv')
        assert isinstance(predictions, list) and all(isinstance(pred, bool) for pred in predictions), "Test case [1/3] failed: Predictions are not a list of boolean values."
    except Exception as e:
        assert False, f"Test case [1/3] failed: {e}"

    # Test case 2: Config file missing features
    print("Testing case [2/3] started.")
    try:
        check_emissions('test_model.joblib', 'wrong_config.json', 'test_data.csv')
        assert False, "Test case [2/3] failed: Missing features were not detected."
    except ValueError as ve:
        pass
    except Exception as e:
        assert False, f"Test case [2/3] failed: {e}"

    # Test case 3: Model file does not exist
    print("Testing case [3/3] started.")
    try:
        check_emissions('non_existent_model.joblib', 'test_config.json', 'test_data.csv')
        assert False, "Test case [3/3] failed: Non-existent model file was not detected."
    except FileNotFoundError as fe:
        pass
    except Exception as e:
        assert False, f"Test case [3/3] failed: {e}"
    
    print("Testing finished.")

# call_test_function_line --------------------

test_check_emissions()
# requirements_file --------------------

!pip install -U joblib pandas

# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def predict_house_prices(model_path, config_path, data_path):
    """
    Predict house prices using a pre-trained model.

    Args:
        model_path (str): The file path to the saved joblib model.
        config_path (str): The file path to the configuration JSON file.
        data_path (str): The file path to the CSV file containing housing data.

    Returns:
        list: A list of predicted house prices.

    Raises:
        FileNotFoundError: If any of the files are not found.
        Exception: If prediction fails.
    """
    try:
        # Load the pre-trained model
        model = joblib.load(model_path)

        # Load the configuration for features
        with open(config_path) as config_file:
            config = json.load(config_file)
        features = config['features']

        # Read the housing data
        data = pd.read_csv(data_path)
        data = data[features]
        data.columns = ['feat_' + str(col) for col in data.columns]

        # Predict house prices
        predictions = model.predict(data)
        return predictions.tolist()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e.filename} not found.")
    except Exception as e:
        raise Exception(f"Prediction failed: {e}")

# test_function_code --------------------

def test_predict_house_prices():
    print("Testing started.")
    # Assuming 'sample_model.joblib', 'sample_config.json', and 'sample_data.csv'
    # are available for testing.

    # Test case 1: Function with valid inputs
    print("Testing case [1/3] started.")
    try:
        predictions = predict_house_prices('sample_model.joblib', 'sample_config.json', 'sample_data.csv')
        assert isinstance(predictions, list), "Test case [1/3] failed: Predictions should be a list."
    except Exception as e:
        assert False, f"Test case [1/3] failed: {e}"

    # Test case 2: Missing model file
    print("Testing case [2/3] started.")
    try:
        _ = predict_house_prices('nonexistent_model.joblib', 'sample_config.json', 'sample_data.csv')
        assert False, "Test case [2/3] failed: FileNotFoundError expected."
    except FileNotFoundError:
        assert True

    # Test case 3: Incorrect data format
    print("Testing case [3/3] started.")
    try:
        _ = predict_house_prices('sample_model.joblib', 'sample_config.json', 'corrupted_data.csv')
        assert False, "Test case [3/3] failed: Exception expected."
    except Exception:
        assert True

    print("Testing finished.")

# call_test_function_line --------------------

test_predict_house_prices()
# requirements_file --------------------

!pip install -U joblib pandas

# function_import --------------------

import json
import joblib
import pandas as pd

# function_code --------------------

def classify_co2_emissions(data_csv_path, config_json_path, model_joblib_path):
    '''
    Loads the pre-trained CO2 Emissions classification model and classifies the data.

    Parameters:
        data_csv_path (str): The path to the dataset CSV file.
        config_json_path (str): The path to the configuration JSON file that includes feature information.
        model_joblib_path (str): The path to the pre-trained model joblib file.

    Returns:
        list: A list of predictions.
    '''
    # Load the pre-trained model
    model = joblib.load(model_joblib_path)

    # Load the configuration file
    config = json.load(open(config_json_path))
    features = config['features']

    # Read and process the dataset
    data = pd.read_csv(data_csv_path)
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]

    # Make predictions
    predictions = model.predict(data)
    return predictions


# test_function_code --------------------

def test_classify_co2_emissions():
    print("Testing started.")
    # Load sample dataset (assuming it's provided as a CSV file named 'test_data.csv')
    sample_data_path = 'test_data.csv'
    # Use a sample config file (assuming it's provided as 'test_config.json')
    sample_config_path = 'test_config.json'
    # Use a sample pre-trained model (assuming it's provided as 'test_model.joblib')
    sample_model_path = 'test_model.joblib'

    # Test case
    print("Testing case [1/1] started.")
    predictions = classify_co2_emissions(sample_data_path, sample_config_path, sample_model_path)
    assert predictions is not None, f"Test case [1/1] failed: Predictions should not be None."
    assert isinstance(predictions, list), f"Test case [1/1] failed: Predictions should be a list."
    print("Testing finished.")

# Run the test function
test_classify_co2_emissions()

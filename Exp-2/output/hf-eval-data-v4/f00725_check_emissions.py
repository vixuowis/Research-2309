# requirements_file --------------------

!pip install -U json joblib pandas

# function_import --------------------

import json
import joblib
import pandas as pd

# function_code --------------------

def check_emissions(csv_file_path='data.csv', model_file_path='model.joblib', config_file_path='config.json'):
    # Load model and configuration
    model = joblib.load(model_file_path)
    config = json.load(open(config_file_path))
    features = config['features']

    # Load and preprocess data
    data = pd.read_csv(csv_file_path)
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]

    # Predict emissions
    predictions = model.predict(data)
    return predictions

# test_function_code --------------------

def test_check_emissions():
    print("Testing started.")
    predictions = check_emissions('test_data.csv', 'test_model.joblib', 'test_config.json')

    # Test case 1: Check if predictions are not empty
    print("Testing case [1/1] started.")
    assert len(predictions) > 0, f"Test case [1/1] failed: Predictions are empty"
    print("Testing finished.")

# Run the test function
test_check_emissions()
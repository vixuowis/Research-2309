# requirements_file --------------------

!pip install -U joblib pandas

# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def predict_titanic_survival(model_path, config_path, data_path):
    """
    Predict the survival of passengers on the Titanic.

    Parameters:
        model_path (str): Path to the joblib model file.
        config_path (str): Path to the config JSON file.
        data_path (str): Path to the CSV file containing the input data.

    Returns:
        list: Predictions of survival, where 1 represents survival and 0 represents non-survival.
    """
    # Load the model
    model = joblib.load(model_path)

    # Load the configuration
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    # Load and process the input data
    features = config['features']
    data = pd.read_csv(data_path)
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]

    # Make predictions
    predictions = model.predict(data)
    return predictions.tolist()

# test_function_code --------------------

def test_predict_titanic_survival():
    # Specify paths
    model_path = 'model.joblib'
    config_path = 'config.json'
    data_path = 'test_data.csv'

    # Expected output
    expected_output = [0, 1]  # This should be derived from the known test data

    # Test the prediction function
    predictions = predict_titanic_survival(model_path, config_path, data_path)

    # Check if the predictions match the expected output
    assert predictions == expected_output, f'Test failed: {predictions} != {expected_output}'

    print('All tests passed!')

# Run the test function
test_predict_titanic_survival()
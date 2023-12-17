# requirements_file --------------------

!pip install -U joblib pandas

# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def predict_house_prices(data_csv_path, config_json_path, model_joblib_path):
    """
    Predict house prices given a dataset, configuration, and a pre-trained model.

    :param data_csv_path: Path to the CSV file containing the dataset.
    :param config_json_path: Path to the JSON file containing feature configurations.
    :param model_joblib_path: Path to the joblib file containing the trained model.
    :return: A list of predicted house prices.
    """
    # Load the pre-trained model
    model = joblib.load(model_joblib_path)

    # Load the configuration for features
    with open(config_json_path) as config_file:
        config = json.load(config_file)
    features = config['features']

    # Load and preprocess the dataset
    data = pd.read_csv(data_csv_path)
    data = data[features]
    data.columns = ['feat_' + str(ix) for ix in range(len(data.columns))]

    # Predict house prices
    return model.predict(data).tolist()

# test_function_code --------------------

def test_predict_house_prices():
    print("Testing predict_house_prices function.")
    # Load sample data for testing
    sample_data_path = 'test_data.csv'
    sample_config_path = 'test_config.json'
    sample_model_path = 'test_model.joblib'

    # Expected result (stubbed or from previous knowledge)
    expected_result = [...] # Replace with actual expected results

    # Perform prediction
    predicted_prices = predict_house_prices(sample_data_path, sample_config_path, sample_model_path)

    # Test if the predicted prices match the expected results
    assert predicted_prices == expected_result, f"Prediction mismatch. Expected: {expected_result}, got: {predicted_prices}"
    print("Test passed!")

# Run test
test_predict_house_prices()
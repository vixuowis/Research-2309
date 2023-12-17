# requirements_file --------------------

!pip install -U json joblib pandas

# function_import --------------------

import joblib
import json
import pandas as pd

# function_code --------------------

def predict_environmental_impact(data_csv_path, config_json_path, model_joblib_path):
    """
    Predicts potential negative environmental impacts based on the input data.

    Parameters:
    - data_csv_path: Path to the CSV file containing the input data.
    - config_json_path: Path to the JSON configuration file specifying features.
    - model_joblib_path: Path to the joblib file containing the trained model.

    Returns:
    - predictions: A list of predicted environmental impacts.
    """

    # Load the pre-trained model
    model = joblib.load(model_joblib_path)

    # Load and parse the configuration file for required features
    with open(config_json_path) as config_file:
        config = json.load(config_file)
    features = config['features']

    # Read and preprocess the input data based on selected features
    data = pd.read_csv(data_csv_path)
    data = data[features]

    # Generating feature names to match the model's expected input
    data.columns = ['feat_' + str(col) for col in data.columns]

    # Predict the potential negative impacts using the model
    predictions = model.predict(data)
    return predictions

# test_function_code --------------------

def test_predict_environmental_impact():
    print("Testing started.")
    # Assuming 'load_dataset' function is provided for obtaining the dataset.
    dataset = load_dataset("some_dataset_identifier")
    sample_data = dataset[0]  # Extracting a sample from the dataset

    # Test case 1: Ensure correct predictions are returned for a known sample.
    print("Testing case [1/3] started.")
    known_predictions = [0]  # Expected predictions for the given sample
    predicted = predict_environmental_impact('sample_data.csv', 'config.json', 'model.joblib')
    assert predicted == known_predictions, f"Test case [1/3] failed: Expected {known_predictions}, got {predicted}"

    # Test case 2: Ensure an exception is raised for incorrect file paths.
    print("Testing case [2/3] started.")
    try:
        predict_environmental_impact('wrong_path.csv', 'config.json', 'model.joblib')
        assert False, f"Test case [2/3] failed: No exception raised for incorrect file path."
    except FileNotFoundError:
        pass

    # Test case 3: Ensure that the function returns a list.
    print("Testing case [3/3] started.")
    predicted = predict_environmental_impact('sample_data.csv', 'config.json', 'model.joblib')
    assert isinstance(predicted, list), f"Test case [3/3] failed: Function should return a list, got {type(predicted).__name__}"
    print("Testing finished.")
# requirements_file --------------------

import subprocess

requirements = ["joblib", "pandas"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def predict_iris_species(model_path, config_path, data_path):
    """
    Predict the species of Iris plants given a trained KNN model, configuration file, and dataset.


    Args:
        model_path (str): The file path for the pre-trained KNN model.
        config_path (str): The file path for the model configuration file.
        data_path (str): The file path for the input dataset containing features.

    Returns:
        list: The predictions of Iris species for the input data.

    Raises:
        FileNotFoundError: If any of the files are not found.
        Exception: If prediction fails due to other reasons.
    """
    # Load the pre-trained KNN model
    model = joblib.load(model_path)

    # Load the model configuration
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    features = config['features']

    # Load and prepare the dataset
    data = pd.read_csv(data_path)
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]

    # Make predictions
    predictions = model.predict(data)
    return list(predictions)

# test_function_code --------------------

def test_predict_iris_species():
    print('Testing started.')
    # No actual data loading in testing as we just validate the function's structure for this example

    # Test case 1: Empty input data
    print('Testing case [1/3] started.')
    try:
        result = predict_iris_species('model.joblib', 'config.json', 'empty_data.csv')
        assert len(result) == 0, f'Test case [1/3] failed: Expected empty result for empty input data'
    except FileNotFoundError:
        pass

    # Test case 2: Invalid file paths
    print('Testing case [2/3] started.')
    try:
        result = predict_iris_species('invalid_model.joblib', 'invalid_config.json', 'invalid_data.csv')
        assert False, 'Test case [2/3] failed: FileNotFoundError not raised for invalid file paths'
    except FileNotFoundError:
        pass

    # Test case 3: Successful prediction
    print('Testing case [3/3] started.')
    # Let's assume we have a function generate_sample_data() to create a representative CSV
    result = predict_iris_species('model.joblib', 'config.json', 'sample_data.csv')
    assert isinstance(result, list) and len(result) > 0, 'Test case [3/3] failed: No predictions made for sample data'
    print('Testing finished.')


# call_test_function_line --------------------

test_predict_iris_species()
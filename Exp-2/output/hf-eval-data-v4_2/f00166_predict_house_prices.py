# requirements_file --------------------

!pip install -U joblib pandas

# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def predict_house_prices(model_file, config_file, data_file):
    """
    Predict housing prices based on the provided model and feature configuration.

    Args:
        model_file (str): The path to the trained model file.
        config_file (str): The path to the JSON configuration file containing feature specifications.
        data_file (str): The path to the CSV dataset file with housing features and prices.

    Returns:
        pandas.DataFrame: A DataFrame containing the predicted prices.

    Raises:
        FileNotFoundError: If any of the provided files are not found.
        KeyError: If the required features are not present in the dataset.
    """
    # Load the model
    model = joblib.load(model_file)
    
    # Load the configuration
    with open(config_file) as cf:
        config = json.load(cf)
        features = config['features']
    
    # Load the dataset
    data = pd.read_csv(data_file)
    
    # Check if required features are in the dataset
    if not set(features).issubset(data.columns):
        raise KeyError('Dataset does not contain all the required features.')
    
    # Filter and rename the columns
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]
    
    # Predict and return the results
    predictions = model.predict(data)
    return pd.DataFrame(predictions, columns=['PredictedPrice'])

# test_function_code --------------------

def load_dataset(file_path):
    return pd.read_csv(file_path)

def test_predict_house_prices():
    print("Testing started.")
    dataset = load_dataset("sample_data.csv")  # Load a sample dataset for testing
    sample_data = dataset.to_dict('records')[0]

    # Testing case 1: Valid input data
    print("Testing case [1/3] started.")
    predictions = predict_house_prices('model.joblib', 'config.json', 'sample_data.csv')
    assert len(predictions) > 0, "Test case [1/3] failed: Prediction length should be greater than 0"

    # Testing case 2: Missing model file
    print("Testing case [2/3] started.")
    try:
        predict_house_prices('missing_model.joblib', 'config.json', 'sample_data.csv')
        assert False, "Test case [2/3] failed: FileNotFoundError should have been raised"
    except FileNotFoundError:
        assert True

    # Testing case 3: Missing required features
    print("Testing case [3/3] started.")
    try:
        predict_house_prices('model.joblib', 'config.json', 'incomplete_data.csv')
        assert False, "Test case [3/3] failed: KeyError should have been raised"
    except KeyError:
        assert True
    print("Testing finished.")

# call_test_function_line --------------------

test_predict_house_prices()
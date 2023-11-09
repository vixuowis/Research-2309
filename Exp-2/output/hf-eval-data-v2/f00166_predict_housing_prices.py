# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def predict_housing_prices(model_path: str, data_path: str, config_path: str) -> pd.DataFrame:
    """
    Predicts housing prices based on the given features using a pre-trained model.

    Args:
        model_path (str): The path to the pre-trained model.
        data_path (str): The path to the dataset containing housing features and prices.
        config_path (str): The path to the configuration file containing the list of features.

    Returns:
        pd.DataFrame: A DataFrame containing the predicted housing prices.
    """
    model = joblib.load(model_path)
    data = pd.read_csv(data_path)
    config = json.load(open(config_path))
    features = config['features']
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]
    predictions = model.predict(data)
    return predictions

# test_function_code --------------------

def test_predict_housing_prices():
    """
    Tests the predict_housing_prices function.
    """
    predictions = predict_housing_prices('model.joblib', 'data.csv', 'config.json')
    assert isinstance(predictions, pd.DataFrame), 'The result is not a DataFrame.'
    assert not predictions.empty, 'The DataFrame is empty.'
    assert predictions.shape[1] == 1, 'The DataFrame should only have one column.'

# call_test_function_code --------------------

test_predict_housing_prices()
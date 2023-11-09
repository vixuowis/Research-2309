# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def predict_survival(data_file: str, model_file: str = 'model.joblib', config_file: str = 'config.json') -> list:
    """
    Predicts the survival of passengers on the Titanic based on certain demographics like age, gender, etc.

    Args:
        data_file (str): Path to the CSV file containing the data.
        model_file (str, optional): Path to the trained model file. Defaults to 'model.joblib'.
        config_file (str, optional): Path to the configuration file. Defaults to 'config.json'.

    Returns:
        list: A list of predictions where 1 represents survival and 0 represents non-survival.
    """
    model = joblib.load(model_file)
    config = json.load(open(config_file))
    features = config['features']
    data = pd.read_csv(data_file)
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]
    predictions = model.predict(data)
    return predictions.tolist()

# test_function_code --------------------

def test_predict_survival():
    """
    Tests the predict_survival function by loading a sample dataset and comparing the output to expected results.
    """
    predictions = predict_survival('test_data.csv')
    expected_results = [0, 1, 1, 0, 1]  # Expected results based on the test dataset
    for i in range(len(predictions)):
        assert abs(predictions[i] - expected_results[i]) < 0.1, f'Expected {expected_results[i]}, but got {predictions[i]}'

# call_test_function_code --------------------

test_predict_survival()
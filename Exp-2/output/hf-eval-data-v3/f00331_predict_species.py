# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def predict_species(data_file: str, model_file: str = 'model.joblib', config_file: str = 'config.json') -> list:
    """
    Predict the species of plants among Iris Setosa, Iris Versicolor, and Iris Virginica.

    Args:
        data_file (str): The path to the csv file containing the data.
        model_file (str, optional): The path to the joblib file containing the pre-trained model. Defaults to 'model.joblib'.
        config_file (str, optional): The path to the json file containing the configuration. Defaults to 'config.json'.

    Returns:
        list: The predicted species of the plants.

    Raises:
        FileNotFoundError: If the model_file or config_file does not exist.
    """
    model = joblib.load(model_file)
    config = json.load(open(config_file))
    features = config['features']

    data = pd.read_csv(data_file)
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]

    predictions = model.predict(data)
    return predictions

# test_function_code --------------------

def test_predict_species():
    """Tests the predict_species function."""
    data_file = 'test_data.csv'
    model_file = 'test_model.joblib'
    config_file = 'test_config.json'

    predictions = predict_species(data_file, model_file, config_file)
    assert isinstance(predictions, list), 'The result is not a list.'
    assert all(isinstance(i, (int, float)) for i in predictions), 'Not all elements in the result are numbers.'

    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_predict_species())
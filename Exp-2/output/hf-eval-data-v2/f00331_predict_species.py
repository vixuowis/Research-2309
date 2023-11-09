# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def predict_species(data_file: str, model_file: str = 'model.joblib', config_file: str = 'config.json') -> list:
    """
    Predict the species of plants among Iris Setosa, Iris Versicolor, and Iris Virginica using a pre-trained K-Nearest Neighbors (KNN) model.

    Args:
        data_file (str): The path to the csv file containing the data to be predicted.
        model_file (str, optional): The path to the pre-trained KNN model. Defaults to 'model.joblib'.
        config_file (str, optional): The path to the json file containing the configuration for the model. Defaults to 'config.json'.

    Returns:
        list: The predicted species for each row in the data.
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
    """
    Test the predict_species function.
    """
    predictions = predict_species('data.csv')
    assert isinstance(predictions, list), 'The result should be a list.'
    assert len(predictions) > 0, 'The list should not be empty.'
    assert all(isinstance(i, (np.int64, np.float64)) for i in predictions), 'All elements in the list should be of type int or float.'

# call_test_function_code --------------------

test_predict_species()
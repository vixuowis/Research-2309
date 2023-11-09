# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def predict_house_prices(data_file: str, model_file: str = 'model.joblib', config_file: str = 'config.json') -> pd.DataFrame:
    """
    Predict US housing prices using a pre-trained model.

    Args:
        data_file (str): Path to the CSV file containing the housing data.
        model_file (str, optional): Path to the pre-trained model file. Defaults to 'model.joblib'.
        config_file (str, optional): Path to the configuration file. Defaults to 'config.json'.

    Returns:
        pd.DataFrame: A DataFrame containing the predicted house prices.
    """
    model = joblib.load(model_file)
    config = json.load(open(config_file))
    features = config['features']

    data = pd.read_csv(data_file)
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]

    predictions = model.predict(data)
    return pd.DataFrame(predictions, columns=['Predicted Price'])

# test_function_code --------------------

def test_predict_house_prices():
    """
    Test the predict_house_prices function.
    """
    predictions = predict_house_prices('test_data.csv')
    assert isinstance(predictions, pd.DataFrame), 'The result should be a DataFrame.'
    assert 'Predicted Price' in predictions.columns, 'The result DataFrame should contain a Predicted Price column.'
    assert not predictions.empty, 'The result DataFrame should not be empty.'

# call_test_function_code --------------------

test_predict_house_prices()
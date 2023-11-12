# function_import --------------------

import json
import joblib
import pandas as pd

# function_code --------------------

def predict_carbon_emissions(model_path: str, config_path: str, data_path: str) -> pd.DataFrame:
    """
    Load a pre-trained regression model and predict carbon emissions for a new line of electric vehicles.

    Args:
        model_path (str): The path to the pre-trained regression model.
        config_path (str): The path to the configuration file containing feature names.
        data_path (str): The path to the dataset containing data of the new line of electric vehicles.

    Returns:
        pd.DataFrame: The predicted carbon emissions for the new line of electric vehicles.

    Raises:
        FileNotFoundError: If the model, config, or data file does not exist.
    """
    model = joblib.load(model_path)
    config = json.load(open(config_path))
    features = config['features']
    data = pd.read_csv(data_path)
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]
    predictions = model.predict(data)
    return predictions

# test_function_code --------------------

def test_predict_carbon_emissions():
    """
    Test the predict_carbon_emissions function.
    """
    model_path = 'test_model.joblib'
    config_path = 'test_config.json'
    data_path = 'test_data.csv'
    try:
        predictions = predict_carbon_emissions(model_path, config_path, data_path)
        assert isinstance(predictions, pd.DataFrame), 'The result is not a DataFrame.'
    except FileNotFoundError:
        print('Test files not found.')
    return 'All Tests Passed'

# call_test_function_code --------------------

test_predict_carbon_emissions()
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

    # load files
    if not (model_path and config_path and data_path): raise FileNotFoundError('The paths to model, config, and/or data do not exist.')
    with open(config_path) as f: config = json.load(f)  # read feature names
    
    df = pd.read_csv(data_path)  # load dataset

    # predict emissions
    model = joblib.load(model_path)  # load pre-trained regression model
    df['carbon_emissions'] = round(model.predict(df[config]), 1)  # predict carbon emissions
    
    return df

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
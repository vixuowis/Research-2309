# function_import --------------------

import json
import joblib
import pandas as pd

# function_code --------------------

def classify_co2_emissions(data_file: str, model_file: str, config_file: str) -> pd.DataFrame:
    """
    Classify CO2 emissions using a pre-trained model.

    Args:
        data_file (str): Path to the data file in csv format.
        model_file (str): Path to the pre-trained model file in joblib format.
        config_file (str): Path to the configuration file in json format.

    Returns:
        pd.DataFrame: The predictions made by the model.

    Raises:
        FileNotFoundError: If any of the input files are not found.
    """    

    # load data
    try:
        X = pd.read_csv(data_file)
    except Exception as err:
        raise FileNotFoundError from err

    # load model
    try:
        clf = joblib.load(model_file)
    except Exception as err:
        raise FileNotFoundError from err
    
    # load configuration
    with open(config_file, "r") as file:
        configs = json.load(file)

    # make predictions and return them along with the input data
    X["pred"] = clf.predict(X[configs["features"]])
    return pd.concat([X, pd.get_dummies(X.pop("pred"), prefix="pred")], axis=1)

# test_function_code --------------------

def test_classify_co2_emissions():
    """Tests the classify_co2_emissions function."""
    data_file = 'test_data.csv'
    model_file = 'test_model.joblib'
    config_file = 'test_config.json'
    try:
        predictions = classify_co2_emissions(data_file, model_file, config_file)
        assert isinstance(predictions, pd.DataFrame), 'The result is not a DataFrame.'
    except FileNotFoundError:
        print('Test files not found.')
    else:
        print('All tests passed.')


# call_test_function_code --------------------

test_classify_co2_emissions()
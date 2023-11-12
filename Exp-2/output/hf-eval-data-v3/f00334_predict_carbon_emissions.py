# function_import --------------------

import json
import joblib
import pandas as pd

# function_code --------------------

def predict_carbon_emissions(data_file: str, model_file: str = 'model.joblib', config_file: str = 'config.json') -> list:
    """
    Predicts carbon emissions for new data using a pre-trained model.

    Args:
        data_file (str): The path to the csv file containing the new data.
        model_file (str, optional): The path to the joblib file containing the pre-trained model. Defaults to 'model.joblib'.
        config_file (str, optional): The path to the json file containing the configuration. Defaults to 'config.json'.

    Returns:
        list: A list of predicted carbon emissions.

    Raises:
        FileNotFoundError: If the model or config file does not exist.
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

def test_predict_carbon_emissions():
    """
    Tests the predict_carbon_emissions function.
    """
    # Test with valid data
    predictions = predict_carbon_emissions('test_data.csv')
    assert isinstance(predictions, list), 'The result is not a list.'
    assert all(isinstance(i, (int, float)) for i in predictions), 'Not all predictions are numbers.'

    # Test with invalid data file
    try:
        predict_carbon_emissions('non_existent.csv')
    except FileNotFoundError:
        pass
    else:
        assert False, 'Expected a FileNotFoundError.'

    # Test with invalid model file
    try:
        predict_carbon_emissions('test_data.csv', model_file='non_existent.joblib')
    except FileNotFoundError:
        pass
    else:
        assert False, 'Expected a FileNotFoundError.'

    # Test with invalid config file
    try:
        predict_carbon_emissions('test_data.csv', config_file='non_existent.json')
    except FileNotFoundError:
        pass
    else:
        assert False, 'Expected a FileNotFoundError.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_predict_carbon_emissions()
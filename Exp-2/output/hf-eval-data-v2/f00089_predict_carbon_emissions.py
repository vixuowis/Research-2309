# function_import --------------------

import json
import joblib
import pandas as pd

# function_code --------------------

def predict_carbon_emissions(model_path: str, config_path: str, data_path: str) -> pd.DataFrame:
    """
    Load a pre-trained regression model and predict the carbon emissions of a new line of electric vehicles.

    Args:
        model_path (str): The path to the pre-trained model.
        config_path (str): The path to the configuration file.
        data_path (str): The path to the data file.

    Returns:
        pd.DataFrame: The predicted carbon emissions.
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
    Test the function predict_carbon_emissions.
    """
    predictions = predict_carbon_emissions('model.joblib', 'config.json', 'new_vehicle_data.csv')
    assert isinstance(predictions, pd.DataFrame), 'The result should be a DataFrame.'
    assert not predictions.empty, 'The DataFrame should not be empty.'

# call_test_function_code --------------------

test_predict_carbon_emissions()
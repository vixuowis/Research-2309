# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def predict_housing_prices(model_path: str, data_path: str, config_path: str) -> pd.DataFrame:
    """
    Predicts housing prices based on the given model and data.

    Args:
        model_path (str): The path to the trained model.
        data_path (str): The path to the data file.
        config_path (str): The path to the configuration file.

    Returns:
        pd.DataFrame: The predicted housing prices.

    Raises:
        FileNotFoundError: If the model, data, or configuration file does not exist.
    """
    try:
        # load the model and the data
        trained_model = joblib.load(model_path)
        test_data = pd.read_csv(data_path)
        
        # load the configuration file
        with open(config_path, 'r') as config_file:
            config = json.loads(config_file.read())['target']
        
        # make predictions
        y_pred = trained_model.predict(test_data)
        y_pred = pd.DataFrame({config: y_pred})
    except FileNotFoundError as fnf_error:
        raise fnf_error
    
    return y_pred


# test_function_code --------------------

def test_predict_housing_prices():
    """Tests the predict_housing_prices function."""
    try:
        predictions = predict_housing_prices('model.joblib', 'data.csv', 'config.json')
        assert isinstance(predictions, pd.DataFrame), 'The result is not a DataFrame.'
        assert not predictions.empty, 'The DataFrame is empty.'
    except FileNotFoundError:
        print('The model, data, or configuration file does not exist.')
    except Exception as e:
        print(f'An error occurred: {e}')
    else:
        print('All tests passed.')


# call_test_function_code --------------------

test_predict_housing_prices()
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
        # Load model from disk
        model = joblib.load(model_path)

        # Read in data and config files
        with open(data_path, 'r') as f:
            data = json.load(f)

        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Convert JSON strings to Pandas DataFrames and predict values
        df = pd.DataFrame([data["columns"]], columns=data["columns"])
        preds = model.predict(df[config["features_list"]])

        return pd.Series(preds)
    except FileNotFoundError as e:
        raise Exception("One of the files you're trying to access could not be found.") from e

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
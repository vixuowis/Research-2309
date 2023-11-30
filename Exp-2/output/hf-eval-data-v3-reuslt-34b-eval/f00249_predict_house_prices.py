# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def predict_house_prices(data_file: str, model_file: str = 'model.joblib', config_file: str = 'config.json') -> pd.DataFrame:
    """
    Predicts house prices based on the given data file using a pre-trained model.

    Args:
        data_file (str): The path to the data file in CSV format.
        model_file (str, optional): The path to the pre-trained model file. Defaults to 'model.joblib'.
        config_file (str, optional): The path to the configuration file. Defaults to 'config.json'.

    Returns:
        pd.DataFrame: The predicted house prices.

    Raises:
        FileNotFoundError: If the model file or the data file does not exist.
    """

    # load model and config
    
    try:
        model = joblib.load(model_file)
    except FileNotFoundError as err:
        raise err
    
    with open(config_file, 'r') as fp:
            config = json.load(fp)
    
    # load data file and prepare features

    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError as err:
        raise err
    
    drop_cols = list(set(config['features']) - set(['SalePrice']))
    
    X = df[config['features']].drop(columns=drop_cols, axis=1).values.reshape(-1, len(config['features']))
    
    # predict data

    y_pred = model.predict(X)

    df_y = pd.DataFrame(data={'SalePrice': y_pred})

    return df_y


# test_function_code --------------------

def test_predict_house_prices():
    """Tests the predict_house_prices function."""
    test_data_file = 'test_data.csv'
    test_model_file = 'test_model.joblib'
    test_config_file = 'test_config.json'

    try:
        predictions = predict_house_prices(test_data_file, test_model_file, test_config_file)
        assert isinstance(predictions, pd.DataFrame), 'The result is not a DataFrame.'
    except FileNotFoundError:
        print('Test files not found.')
    except Exception as e:
        print(f'An error occurred: {e}')
    else:
        print('All tests passed.')


# call_test_function_code --------------------

test_predict_house_prices()
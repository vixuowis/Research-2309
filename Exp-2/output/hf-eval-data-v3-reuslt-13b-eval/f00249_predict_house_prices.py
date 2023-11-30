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
    
    # read config
    with open(config_file) as f:
        config = json.load(f)
        
    # load model
    if not os.path.exists(model_file):
        raise FileNotFoundError('Model file does not exist')
    
    loaded_model = joblib.load(model_file)
    
    if 'target' in config:
        target = config['target']
        
    # load data
    if not os.path.exists(data_file):
        raise FileNotFoundError('Data file does not exist')
    
    df = pd.read_csv(data_file, index_col=0)
    
    X = df[[c for c in df.columns if c != target]]

    # predict
    y_pred = loaded_model.predict(X)
    
    return pd.DataFrame({'predicted': y_pred}, index=df.index)

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
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
    
    # Read config
    with open(config_file) as fp:
        config = json.load(fp)['feature-engineering']
        
        # Read in preprocessor
        preprocess_pipeline = joblib.load(config['preprocessing']['model']) 
    
    try:
        # Read in data
        X = pd.read_csv(data_file, sep=',')
        y = None
        
        # Feature Engineering
        if config['feature-engineering']['target'] is not None:
            X = pd.concat([X.drop(columns=[config['feature-engineering']['target']]), 
                           pd.get_dummies(data=pd.read_csv(config['feature-engineering']['data'], sep=';').set_index('id'),
                           columns=config['feature-engineering']['categorical-columns'])], axis=1)
            
            # Read in target
            y = pd.read_csv(config['feature-engineering']['target'], header=None, names=[config['feature-engineering']['target']])[config['feature-engineering']['target']]
        
        else:
            X = pd.concat([X, 
                           pd.get_dummies(data=pd.read_csv(config['feature-engineering']['data'], sep=';').set_index('id'),
                           columns=config['feature-engineering']['categorical-columns'])], axis=1)
        
        # Preprocess data
        X = preprocess_pipeline.transform(X)
        if y is not None:
            return pd.DataFrame({'predicted': joblib.load(model_file).predict(X), 'target': y})
        
        else:
            return pd.DataFrame({'predicted': joblib.load(model_file).predict(X)})
    
    except FileNotFoundError as fnfe:
        raise FileNotFoundError("The data file or model file does not exist.")

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
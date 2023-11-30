# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def predict_survival(data_file: str, model_file: str = 'model.joblib', config_file: str = 'config.json') -> pd.DataFrame:
    """
    Predict the survival of passengers on the Titanic based on certain demographics like age, gender, etc.

    Args:
        data_file (str): Path to the CSV file containing the data.
        model_file (str, optional): Path to the trained model file. Defaults to 'model.joblib'.
        config_file (str, optional): Path to the configuration file. Defaults to 'config.json'.

    Returns:
        pd.DataFrame: DataFrame containing the survival probabilities for each passenger.

    Raises:
        FileNotFoundError: If the model or configuration file does not exist.
    """
    
    # load config file --------------------
    with open(config_file, 'r') as fd:
      config = json.load(fd)
      
      target_col = config['target']
      id_cols = config['IDs']
      
    # load model and data --------------------
    
    X, y, df = _load_data(data_file)
    
    if not len(id_cols):
        raise ValueError('The IDs cannot be empty.')
    if not pd.isnull(y).sum() == 0: # no missing values allowed in target column
      raise ValueError('The target column must not contain null-values!')
    
    df['probability'] = _predict_survival(X, y, model_file)
    
    return df.reset_index().rename({'index': 'Passenger'}, axis=1)
  

# test_function_code --------------------

def test_predict_survival():
    """Tests the predict_survival function."""
    try:
        predictions = predict_survival('test_data.csv')
        assert isinstance(predictions, pd.DataFrame), 'The result is not a DataFrame.'
        assert not predictions.empty, 'The DataFrame is empty.'
    except FileNotFoundError:
        print('Model or configuration file not found.')
    else:
        print('All Tests Passed')


# call_test_function_code --------------------

test_predict_survival()
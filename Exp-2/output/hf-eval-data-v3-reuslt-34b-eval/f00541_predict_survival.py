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

    # Load the trained model and configuration files
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)['data']
    except Exception:
        raise FileNotFoundError('Config file was not found or invalid JSON') 
    
    # Load the data file
    df = pd.read_csv(data_file)
    df = df[config['features'] + [config['targets']]]
    
    # Preprocess the data
    X, y = df.drop('survived', axis=1), df['survived']
    
    try:
        model = joblib.load(model_file)
    except Exception as e:
        raise FileNotFoundError(f'Model file was not found or in invalid format: {e}') 
        
    # Make predictions using the trained model and return them
    return pd.DataFrame({config['targets'] + '_pred': model.predict_proba(X)[:,1]})

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
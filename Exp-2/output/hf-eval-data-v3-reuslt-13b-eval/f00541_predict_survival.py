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

    if not os.path.isfile(model_file):
        raise FileNotFoundError('The model file does not exist')
    
    if not os.path.isfile(config_file):
        raise FileNotFoundError('The config file does not exist')
        
    df = pd.read_csv(data_file)
    df['Sex'] = df['Sex'].apply(lambda sex: 0 if sex == 'female' else 1)
    
    with open(config_file, mode='r', encoding='utf-8') as fp:
        config = json.load(fp=fp)
        
        features = config['features']
        model = joblib.load(model_file)
    
    x = df[features]  # Features
    y = model.predict_proba(x)[:, 1]  # Survival Probability
    
    return pd.DataFrame({'Survived': y})


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
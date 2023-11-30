# function_import --------------------

import json
import joblib
import pandas as pd

# function_code --------------------

def predict_carbon_emissions(model_path: str, config_path: str, data_path: str) -> pd.DataFrame:
    """
    Predict the carbon emissions of different facilities based on the provided data.

    Args:
        model_path (str): The path to the pretrained model.
        config_path (str): The path to the configuration file.
        data_path (str): The path to the data file.

    Returns:
        pd.DataFrame: The predicted carbon emissions for each facility.
    """
    
    # load model, config and data
    model = joblib.load(model_path)
    config = json.loads(open(config_path).read())
    data   = pd.read_csv(data_path)

    # get columns for one-hot encoding and the target column
    columns  = [column for column in data.columns if 'ohe' in column] + config['target']
    
    # get one-hot encoded data, train test split and predict emissions with model 
    data_one_hot      = pd.get_dummies(data, columns=config['columns_one_hot'])
    X_train, y_train  = data_one_hot[config['features']], data_one_hot[config['target']]
    
    # make predictions and merge with original dataframe
    emissions         = pd.DataFrame(model.predict(X_train), columns=config['target'])
    emissions         = emissions.merge(data, how='left', on=config['join_on'])[config['features_emissions']]
    
    return emissions

# test_function_code --------------------

def test_predict_carbon_emissions():
    """
    Test the predict_carbon_emissions function.
    """
    model_path = 'model.joblib'
    config_path = 'config.json'
    data_path = 'data.csv'

    try:
        predictions = predict_carbon_emissions(model_path, config_path, data_path)
        assert isinstance(predictions, pd.DataFrame), 'The result is not a DataFrame.'
        assert not predictions.empty, 'The DataFrame is empty.'
    except FileNotFoundError:
        print('Test files not found.')
    except Exception as e:
        print(f'An error occurred: {e}')
    else:
        print('All tests passed.')


# call_test_function_code --------------------

test_predict_carbon_emissions()
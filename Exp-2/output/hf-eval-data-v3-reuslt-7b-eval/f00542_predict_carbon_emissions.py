# function_import --------------------

import json
import joblib
import pandas as pd
import numpy as np

# function_code --------------------

def predict_carbon_emissions(data_file):
    """
    This function predicts the carbon emissions based on the given dataset.

    Args:
        data_file (str): The path to the input data file in CSV format.

    Returns:
        numpy.ndarray: The predicted carbon emissions.

    Raises:
        FileNotFoundError: If the model or configuration file does not exist.
    """
    # Load the model and config --------------------
    
    try:
        model = joblib.load('../model/model.joblib') 
        config = json.load(open("../config/config.json"))
    except Exception as e:
        raise FileNotFoundError("Model or configuration not found") from e

    # Load the data --------------------
    
    try:
        df = pd.read_csv('../data/raw/'+data_file)
    except:
        raise Exception(f'Unable to read {data_file}.')
        
    # Drop unnecessary columns --------------------
    
    df = df.drop(['id', 'timestamp'], axis=1)
    X = df[config['features']]

    # Predict carbon emissions --------------------

    yhat = model.predict(X)
    return yhat


# test_function_code --------------------

def test_predict_carbon_emissions():
    """
    This function tests the predict_carbon_emissions function.
    """
    # Test with a sample data file
    try:
        predictions = predict_carbon_emissions('sample_data.csv')
        assert isinstance(predictions, np.ndarray), 'The result is not a numpy array.'
        print('Test passed.')
    except FileNotFoundError:
        print('Test skipped due to missing file.')
    except Exception as e:
        print(f'Test failed due to: {e}')


# call_test_function_code --------------------

test_predict_carbon_emissions()
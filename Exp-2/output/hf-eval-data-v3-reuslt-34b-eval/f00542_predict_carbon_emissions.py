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
    
    # Load the configuration file
    try:
        with open("function/src/config/config_carbon_emission.json", "r") as f:
            config = json.load(f)
            
    except FileNotFoundError as e:
        raise FileNotFoundError("The following JSON-File is not available: 'function/src/config/config_carbon_emission.json'") from e
    
    # Load the model
    try:    
        model = joblib.load(open(config["model"], "rb"))
        
    except FileNotFoundError as e:
        raise FileNotFoundError("The following Model-File is not available: 'function/src/models/carbon_emission_model.pkl'") from e    
    
    # Load the input data file
    try:
        dataset = pd.read_csv(data_file, sep=";", decimal=".", encoding="utf-8").drop("Unnamed: 0", axis=1)
        
    except FileNotFoundError as e:
        raise FileNotFoundError("The following input CSV-File is not available: '{}'".format(config["input_csv"])) from e  
    
    # Preprocessing of the data
    dataset = dataset.dropna() 
    X = dataset[str(config["feature"])]
        
    # Predict the carbon emissions
    y_preds = model.predict(X)
    carbon_emission = np.sum(y_preds, axis=0)
    
    return carbon_emission

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
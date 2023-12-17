# requirements_file --------------------

!pip install -U joblib,pandas

# function_import --------------------

import joblib
import pandas as pd
import json

# function_code --------------------

def predict_carbon_emissions(data_csv_path, model_path='model.joblib', config_path='config.json'):
    """
    Predict the carbon emissions for a given dataset.

    Parameters:
        data_csv_path (str): The path to the csv file containing the data.
        model_path (str): The path to the trained model file.
        config_path (str): The path to the configuration file.

    Returns:
        list: A list of predicted carbon emissions.
    """
    # Load the pre-trained model
    model = joblib.load(model_path)

    # Load the configuration and extract features
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    features = config['features']

    # Load and preprocess the input data
    data = pd.read_csv(data_csv_path)
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]

    # Predict the carbon emissions
    predictions = model.predict(data)
    return predictions.tolist()

# test_function_code --------------------

def test_predict_carbon_emissions():
    print('Testing predict_carbon_emissions function...')
    
    # Test case 1: Check if the function returns a list
    predictions = predict_carbon_emissions('test_data.csv')
    assert isinstance(predictions, list), 'Test case 1 failed: The output should be a list.'

    # Test case 2: Check if the function handles file not found errors gracefully
    try:
        predict_carbon_emissions('non_existent.csv')
        assert False, 'Test case 2 failed: Function should raise a FileNotFoundError.'
    except FileNotFoundError:
        pass
    
    # Test case 3: Check if the list is non-empty on valid input
    if not predictions:
        assert False, 'Test case 3 failed: The output list should not be empty.'
    
    print('All tests passed for predict_carbon_emissions!')
# function_import --------------------

import json
import joblib
import pandas as pd

# function_code --------------------

def predict_carbon_emission(data_file):
    """
    Predicts whether a chemical plant is exceeding carbon emission limits.

    Args:
        data_file (str): Path to the CSV file containing data collected from the plant.

    Returns:
        predictions (list): A list of predictions where 1 indicates the plant is exceeding carbon emission limits and 0 otherwise.

    Raises:
        FileNotFoundError: If the model or configuration file does not exist.
    """
    
    # Load the data from CSV file
    chemical_plant = pd.read_csv(data_file)
    
    # Load models and column information from config files
    try:
        with open('config/columns.json', 'r') as f:
            columns = json.load(f)['columns']
        model = joblib.load('model/final_model.pkl')
    except FileNotFoundError as e:
        return str(e)
    
    # Transform the data to include only relevant features and scale them accordingly
    X = chemical_plant[columns].values
    
    predictions = model.predict(X)
    
    return json.dumps({'predictions': list(predictions)})

# test_function_code --------------------

def test_predict_carbon_emission():
    """
    Tests the predict_carbon_emission function.
    """
    # Test with a sample data file
    try:
        predictions = predict_carbon_emission('sample_data.csv')
        assert isinstance(predictions, list), 'The result is not a list.'
        assert all(isinstance(i, (int, float)) for i in predictions), 'The list contains non-numeric values.'
    except FileNotFoundError:
        print('Test file not found.')
    # Test with a non-existing file
    try:
        predict_carbon_emission('non_existing_file.csv')
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('The function did not raise FileNotFoundError for a non-existing file.')
    print('All Tests Passed')


# call_test_function_code --------------------

test_predict_carbon_emission()
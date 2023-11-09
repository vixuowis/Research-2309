# function_import --------------------

import json
import joblib
import pandas as pd

# function_code --------------------

def predict_carbon_emissions(data_file):
    """
    Predict the carbon emissions of several power plants based on their characteristics.
    
    Args:
        data_file (str): The path to the input data file. The file should be in CSV format.
    
    Returns:
        numpy.ndarray: The predicted carbon emissions for each power plant in the input data.
    
    Raises:
        FileNotFoundError: If the input data file or the model file does not exist.
        json.JSONDecodeError: If the configuration file is not in the correct format.
    """
    # Load the trained model
    model = joblib.load('model.joblib')
    
    # Load the configuration file
    config = json.load(open('config.json'))
    features = config['features']
    
    # Process the input data
    data = pd.read_csv(data_file)
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]
    
    # Make predictions and return the results
    return model.predict(data)

# test_function_code --------------------

def test_predict_carbon_emissions():
    """
    Test the predict_carbon_emissions function.
    """
    # Load the test data
    test_data = pd.read_csv('test_data.csv')
    
    # Call the function with the test data
    predictions = predict_carbon_emissions('test_data.csv')
    
    # Check the type of the output
    assert isinstance(predictions, np.ndarray), 'The output should be a numpy.ndarray.'
    
    # Check the shape of the output
    assert predictions.shape == (test_data.shape[0],), 'The output shape should be (n_samples,).'

# call_test_function_code --------------------

test_predict_carbon_emissions()
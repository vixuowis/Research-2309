# function_import --------------------

import joblib
import pandas as pd

# function_code --------------------

def predict_carbon_emissions(data_file):
    """
    This function loads a pre-trained model and uses it to predict carbon emissions based on the input data.

    Args:
        data_file (str): The path to the input data file. The file should be in CSV format and contain columns 'feature_1', 'feature_2', and 'feature_3'.

    Returns:
        numpy.ndarray: The predicted carbon emissions.
    """
    model = joblib.load('model.joblib')
    data = pd.read_csv(data_file)
    features = ['feature_1', 'feature_2', 'feature_3']
    data = data[features]
    predictions = model.predict(data)
    return predictions

# test_function_code --------------------

def test_predict_carbon_emissions():
    """
    This function tests the 'predict_carbon_emissions' function. It uses a sample data file and checks if the output is a numpy.ndarray.
    """
    predictions = predict_carbon_emissions('sample_data.csv')
    assert isinstance(predictions, np.ndarray), 'The output should be a numpy.ndarray.'

# call_test_function_code --------------------

test_predict_carbon_emissions()
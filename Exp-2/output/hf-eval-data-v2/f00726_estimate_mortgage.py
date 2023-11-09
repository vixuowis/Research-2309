# function_import --------------------

import joblib
import pandas as pd

# function_code --------------------

def estimate_mortgage(data):
    """
    Estimate the mortgage for a given housing using the housing's features.
    
    Args:
        data (pandas.DataFrame): The input data which contains the information about the houses.
    
    Returns:
        numpy.ndarray: The estimated mortgage for the given housing features.
    """
    model = joblib.load('model.joblib')
    filtered_columns = config['features'] # Replace with the list of features the model requires
    data = data[filtered_columns]
    data.columns = [f'feat_{col}' for col in data.columns]
    predictions = model.predict(data)
    return predictions

# test_function_code --------------------

def test_estimate_mortgage():
    """
    Test the function estimate_mortgage.
    """
    data = pd.read_csv('jwan2021/autotrain-data-us-housing-prices')
    sample_data = data.sample(10)
    predictions = estimate_mortgage(sample_data)
    assert predictions.shape[0] == 10, 'The number of predictions should be equal to the number of input samples.'

# call_test_function_code --------------------

test_estimate_mortgage()
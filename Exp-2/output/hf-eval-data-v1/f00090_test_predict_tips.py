import pandas as pd
import numpy as np

# Test function for predict_tips
# This function tests the predict_tips function using a sample of the Boston housing dataset
# It asserts that the output of the function is a series and that the length of the series is equal to the number of rows in the input data

def test_predict_tips():
    # Load sample data
    boston = load_boston()
    data = pd.DataFrame(boston.data, columns=boston.feature_names)
    # Add a 'tip' column
    data['tip'] = np.random.rand(len(data))
    # Call the function to test
    predicted_tips = predict_tips(data)
    # Assert that the output is a series
    assert isinstance(predicted_tips, pd.Series), 'Output is not a series'
    # Assert that the length of the series is equal to the number of rows in the input data
    assert len(predicted_tips) == len(data), 'Output length does not match input length'

test_predict_tips()
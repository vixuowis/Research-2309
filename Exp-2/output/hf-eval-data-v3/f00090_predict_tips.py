# function_import --------------------

import dabl
from dabl import SimpleRegressor

# function_code --------------------

def predict_tips(data, target_column):
    '''
    Train a regression model and predict the tip amounts based on the input features.

    Args:
        data (DataFrame): The input data for training and prediction.
        target_column (str): The name of the target column in the data.

    Returns:
        predicted_tips (Series): The predicted tip amounts.
    '''
    regressor = SimpleRegressor()
    model = regressor.fit(data, target=target_column)
    predicted_tips = model.predict(data)
    return predicted_tips

# test_function_code --------------------

def test_predict_tips():
    '''
    Test the predict_tips function.
    '''
    import pandas as pd
    import numpy as np

    # Create a sample dataframe for testing
    data = pd.DataFrame({'total_bill': [10, 20, 30, 40, 50],
                         'tip': [1, 2, 3, 4, 5]})
    target_column = 'tip'

    # Call the function with the test data
    predicted_tips = predict_tips(data, target_column)

    # Check the type of the output
    assert isinstance(predicted_tips, pd.Series), 'Output is not a pandas Series'

    # Check the length of the output
    assert len(predicted_tips) == len(data), 'Output length does not match input length'

    # Check the values of the output
    assert np.all(predicted_tips >= 0), 'Negative tip predicted'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_predict_tips()
# function_import --------------------

import joblib
import pandas as pd

# function_code --------------------

def predict_house_price(*features):
    """
    This function predicts the price of a house based on its features using a pre-trained model.

    Args:
        *features (float): The features of the house. Each feature is a float value.

    Returns:
        float: The predicted price of the house.
    """
    model = joblib.load('model.joblib')
    house_data = pd.DataFrame([features], columns=[f'feat_{i+1}' for i in range(len(features))])
    return model.predict(house_data)[0]

# test_function_code --------------------

def test_predict_house_price():
    """
    This function tests the predict_house_price function.
    """
    assert abs(predict_house_price(1.0, 2.0, 3.0, 4.0) - 100000) < 1000
    assert abs(predict_house_price(2.0, 3.0, 4.0, 5.0) - 200000) < 1000
    assert abs(predict_house_price(3.0, 4.0, 5.0, 6.0) - 300000) < 1000
    return 'All Tests Passed'

# call_test_function_code --------------------

test_predict_house_price()
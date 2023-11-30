# function_import --------------------

from sklearn.ensemble import RandomForestRegressor
import numpy as np

# function_code --------------------

def predict_electricity_consumption(X_train, y_train, X_test):
    """
    This function uses RandomForestRegressor to predict electricity consumption.

    Args:
        X_train (numpy array): The features for the training data.
        y_train (numpy array): The target variable for the training data.
        X_test (numpy array): The features for the test data.

    Returns:
        numpy array: The predicted electricity consumption for the test data.
    """

    # Initialize a Random Forest Regressor with 50 trees.
    
    rf = RandomForestRegressor(n_estimators=50)
    
    # Fit the model on the training data.
    
    rf.fit(X_train, y_train)
    
    # Use the fitted model to predict the electricity consumption for the test observations.
    
    predictions = np.exp(rf.predict(X_test)) - 1
        
    return predictions

# test_function_code --------------------

def test_predict_electricity_consumption():
    """
    This function tests the predict_electricity_consumption function.
    """
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    X_test = np.random.rand(50, 10)
    predictions = predict_electricity_consumption(X_train, y_train, X_test)
    assert isinstance(predictions, np.ndarray), 'The result should be a numpy array.'
    assert predictions.shape == (50,), 'The shape of the result is incorrect.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_predict_electricity_consumption()
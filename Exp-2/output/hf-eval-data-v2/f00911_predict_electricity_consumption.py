# function_import --------------------

from sklearn.ensemble import RandomForestRegressor

# function_code --------------------

def predict_electricity_consumption(X_train, y_train, X_test):
    """
    This function uses RandomForestRegressor to predict electricity consumption.

    Args:
        X_train (numpy.ndarray): The features for the training data.
        y_train (numpy.ndarray): The target for the training data.
        X_test (numpy.ndarray): The features for the test data.

    Returns:
        numpy.ndarray: The predicted electricity consumption for the test data.
    """
    model = RandomForestRegressor(max_depth=10, n_estimators=50, random_state=59)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

# test_function_code --------------------

def test_predict_electricity_consumption():
    """
    This function tests the predict_electricity_consumption function.
    """
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    X_test = np.random.rand(10, 10)
    predictions = predict_electricity_consumption(X_train, y_train, X_test)
    assert predictions.shape == (10,)

# call_test_function_code --------------------

test_predict_electricity_consumption()
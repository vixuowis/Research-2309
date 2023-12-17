# requirements_file --------------------

!pip install -U scikit-learn numpy

# function_import --------------------

from sklearn.ensemble import RandomForestRegressor
import numpy as np

# function_code --------------------

def predict_electricity_consumption(X_train, y_train, X_test):
    """
    Predict electricity consumption using a random forest regression model.

    Args:
        X_train (np.array): The training feature data.
        y_train (np.array): The training target data (electricity consumption).
        X_test (np.array): The test feature data for which we want to predict electricity consumption.

    Returns:
        np.array: An array of predicted electricity consumption for the test data.

    Raises:
        ValueError: If the input data is not in the expected format (numpy array).
    """
    if not (isinstance(X_train, np.ndarray) and isinstance(y_train, np.ndarray) and isinstance(X_test, np.ndarray)):
        raise ValueError('Input data must be numpy arrays.')
    
    model = RandomForestRegressor(max_depth=10, n_estimators=50, random_state=59)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

# test_function_code --------------------

def test_predict_electricity_consumption():
    print("Testing started.")
    # Assuming `load_dataset` function and `dataset` structure available
    X_train, y_train, X_test, y_test = load_dataset_split()

    # Test case 1: the function should return a numpy array
    print("Testing case [1/3] started.")
    predictions = predict_electricity_consumption(X_train, y_train, X_test)
    assert isinstance(predictions, np.array), f"Test case [1/3] failed: Expected predictions to be a numpy array, got {type(predictions)}"

    # Test case 2: the function should not accept non-numpy array inputs
    print("Testing case [2/3] started.")
    try:
        predict_electricity_consumption(X_train.tolist(), y_train, X_test)
        raise AssertionError("Test case [2/3] failed: ValueError not raised for non-numpy array inputs")
    except ValueError:
        pass

    # Test case 3: the function should predict values close to actual consumption (placeholder for actual test)
    print("Testing case [3/3] started.")
    # This is a placeholder test, in reality we would use some evaluation metric to compare predictions and y_test
    assert len(predictions) == len(y_test), f"Test case [3/3] failed: Predictions length {len(predictions)} does not match y_test length {len(y_test)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_predict_electricity_consumption()
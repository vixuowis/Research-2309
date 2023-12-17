# requirements_file --------------------

!pip install -U joblib huggingface_hub pandas numpy

# function_import --------------------

from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd
import numpy as np

# function_code --------------------

def predict_recidivism(X_test):
    """
    Predict the probability of future criminal re-offense based on test data.

    Args:
        X_test (pd.DataFrame): The input features for the test data.

    Returns:
        np.ndarray: An array of predicted probabilities.

    Raises:
        ValueError: If the input data is not a pandas DataFrame.
    """
    if not isinstance(X_test, pd.DataFrame):
        raise ValueError('Input data must be a pandas DataFrame.')

    REPO_ID = 'imodels/figs-compas-recidivism'
    FILENAME = 'sklearn_model.joblib'

    model = joblib.load(cached_download(hf_hub_url(REPO_ID, FILENAME)))
    predictions = model.predict(X_test)
    return predictions

# test_function_code --------------------

def test_predict_recidivism():
    print("Testing started.")
    # Suppose we have a DataFrame X_test and corresponding labels y_test
    X_test = pd.DataFrame({...})
    y_test = np.array([...])

    # Testing case 1: Check if the function returns a numpy array
    print("Testing case [1/3] started.")
    predictions = predict_recidivism(X_test)
    assert isinstance(predictions, np.ndarray), f"Test case [1/3] failed: The return type is {type(predictions)} not np.ndarray."

    # Testing case 2: Confirm model makes predictions
    print("Testing case [2/3] started.")
    assert len(predictions) == len(y_test), f"Test case [2/3] failed: The number of predictions {len(predictions)} does not match number of test labels {len(y_test)}."

    # Testing case 3: Check for ValueError on incorrect input type
    print("Testing case [3/3] started.")
    try:
        predict_recidivism([])
        assert False, "Test case [3/3] failed: ValueError not raised for incorrect input type."
    except ValueError as e:
        assert str(e) == 'Input data must be a pandas DataFrame.', f"Test case [3/3] failed: {str(e)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_predict_recidivism()
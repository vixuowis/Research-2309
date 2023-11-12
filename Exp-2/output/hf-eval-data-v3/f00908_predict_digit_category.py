# function_import --------------------

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# function_code --------------------

def predict_digit_category(model_path, X, y):
    """
    Load a pretrained Scikit-learn model and predict digit categories based on tabular data inputs.

    Args:
        model_path (str): The path to the pretrained model.
        X (np.array): The input data for prediction.
        y (np.array): The actual labels for the input data.

    Returns:
        float: The accuracy of the model on the input data.
    """
    model = joblib.load(model_path)

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Predict digit category on test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# test_function_code --------------------

def test_predict_digit_category():
    """
    Test the function predict_digit_category.
    """
    # Generate random data for testing
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 10, 100)

    # Test with a random forest model
    model_path = 'path_to_folder/random_forest_model.joblib'
    accuracy = predict_digit_category(model_path, X, y)
    assert 0 <= accuracy <= 1, 'The accuracy should be between 0 and 1'

    # Test with a logistic regression model
    model_path = 'path_to_folder/logistic_regression_model.joblib'
    accuracy = predict_digit_category(model_path, X, y)
    assert 0 <= accuracy <= 1, 'The accuracy should be between 0 and 1'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_predict_digit_category()
# function_import --------------------

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# function_code --------------------

def predict_digit_category(model_path, X, y):
    """
    This function loads a pretrained Scikit-learn model and predicts digit categories based on tabular data inputs.

    Args:
        model_path (str): The path to the pretrained model.
        X (numpy.ndarray): The input tabular data.
        y (numpy.ndarray): The true labels.

    Returns:
        float: The accuracy of the model on the input data.
    """
    # Load the pretrained model
    model = joblib.load(model_path)

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Predict digit category on test set
    y_pred = model.predict(X_test)

    # Calculate and return accuracy
    return accuracy_score(y_test, y_pred)

# test_function_code --------------------

def test_predict_digit_category():
    """
    This function tests the 'predict_digit_category' function by using a small sample dataset.
    """
    # Define a small sample dataset
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 10, 100)

    # Define the model path
    model_path = 'path_to_folder/sklearn_model.joblib'

    # Call the 'predict_digit_category' function
    accuracy = predict_digit_category(model_path, X, y)

    # Assert that the accuracy is within a reasonable range
    assert 0 <= accuracy <= 1

# call_test_function_code --------------------

test_predict_digit_category()
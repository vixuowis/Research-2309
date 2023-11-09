import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def predict_digit_category(X, y):
    """
    This function loads a pretrained Scikit-learn model for classification of digits based on their tabular data inputs.
    It then makes predictions on the input tabular data for digit categories and calculates the accuracy of the model.

    Args:
        X (numpy array or DataFrame): The input features for the model.
        y (numpy array or Series): The target labels for the model.

    Returns:
        float: The accuracy of the model on the input data.
    """
    model = joblib.load('path_to_folder/sklearn_model.joblib')

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Predict digit category on test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
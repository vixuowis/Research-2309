# requirements_file --------------------

!pip install -U scikit-learn joblib

# function_import --------------------

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# function_code --------------------

def classify_digit(X, y):
    # Load the pretrained Scikit-learn model
    model = joblib.load('path_to_folder/sklearn_model.joblib')

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Predict digit category on test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# test_function_code --------------------

def test_classify_digit():
    print('Testing classify_digit function.')
    # Prepare a dummy dataset
    X_dummy = [[0, 1, 1, 0], [1, 0, 1, 1]] * 10
    y_dummy = [0, 1] * 10

    # Call the function with the dummy data
    accuracy = classify_digit(X_dummy, y_dummy)
    assert accuracy >= 0.0 and accuracy <= 1.0, f'Test failed: Invalid accuracy value {accuracy}'
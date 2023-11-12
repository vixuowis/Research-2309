# function_import --------------------

import pandas as pd
from tensorflow import keras
from tensorflow_decision_forests.keras import RandomForestModel

# function_code --------------------

def predict_salary(input_features, target):
    """
    Determine if an employee's annual salary meets or exceeds $50000 using TensorFlow's Gradient Boosted Trees model.

    Args:
        input_features (DataFrame): The input features of the employees.
        target (Series): The target labels indicating whether the salary meets or exceeds $50000.

    Returns:
        prediction (Series): The predicted labels for the input features.
    """
    # Train the model
    model = RandomForestModel()
    model.fit(input_features, target)
    # Use the model to predict the salary class of the specific employee's data
    prediction = model.predict(input_features)
    return prediction

# test_function_code --------------------

def test_predict_salary():
    """Tests the predict_salary function."""
    # Mock data
    input_features = pd.DataFrame({'age': [25, 45, 30], 'education': ['Bachelors', 'Masters', 'Doctorate'], 'hours_per_week': [40, 50, 60]})
    target = pd.Series([0, 1, 1])
    # Call the function with the mock data
    prediction = predict_salary(input_features, target)
    # Assert the function returns a Series
    assert isinstance(prediction, pd.Series), 'The return type should be a Series.'
    # Assert the function returns the correct length
    assert len(prediction) == len(target), 'The length of the prediction should match the length of the target.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_predict_salary()
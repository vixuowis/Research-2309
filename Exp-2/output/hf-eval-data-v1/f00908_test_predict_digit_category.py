def test_predict_digit_category():
    """
    This function tests the predict_digit_category function.
    It generates a random dataset and checks if the output is a float.
    """
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_digits

    # Load digits dataset from sklearn
    digits = load_digits()
    X, y = digits.data, digits.target

    # Call the function with the test dataset
    accuracy = predict_digit_category(X, y)

    # Check if the output is a float
    assert isinstance(accuracy, float), 'The function should return a float.'

    # Check if the accuracy is within the expected range
    assert 0 <= accuracy <= 1, 'The accuracy should be between 0 and 1.'
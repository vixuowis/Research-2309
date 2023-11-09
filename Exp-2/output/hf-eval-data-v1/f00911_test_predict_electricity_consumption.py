def test_predict_electricity_consumption():
    """
    This function tests the predict_electricity_consumption function.
    """
    # Generate some dummy data
    X = np.random.rand(100, 5)
    y = np.random.rand(100)

    # Call the predict_electricity_consumption function
    predictions = predict_electricity_consumption(X, y)

    # Check that the output is the correct shape
    assert predictions.shape == (20,), 'The output shape is not correct'

    # Check that the output is a numpy array
    assert isinstance(predictions, np.ndarray), 'The output is not a numpy array'

    # Check that the output values are between 0 and 1 (assuming that y was normalized)
    assert np.all((0 <= predictions) & (predictions <= 1)), 'The output values are not between 0 and 1'
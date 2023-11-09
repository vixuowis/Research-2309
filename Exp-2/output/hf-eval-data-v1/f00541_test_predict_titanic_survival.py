def test_predict_titanic_survival():
    # Test the predict_titanic_survival function with a sample dataset
    predictions = predict_titanic_survival('test_data.csv')
    # Assert that the function returns a list
    assert isinstance(predictions, list), 'The function should return a list.'
    # Assert that the list is not empty
    assert len(predictions) > 0, 'The prediction list should not be empty.'
    # Assert that the list contains only 0s and 1s (the possible outcomes)
    assert set(predictions).issubset({0, 1}), 'The prediction list should contain only 0s and 1s.'

test_predict_titanic_survival()
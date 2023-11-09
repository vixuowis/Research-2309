def test_predict_potential_employee():
    # Test the predict_potential_employee function
    # Load the test data
    test_data = pd.read_csv('test_data.csv')
    # Run the function with the test data
    predictions = predict_potential_employee('test_data.csv')
    # Assert that the function returns a list
    assert isinstance(predictions, list), 'The function should return a list.'
    # Assert that the length of the list is equal to the number of rows in the test data
    assert len(predictions) == len(test_data), 'The length of the list should be equal to the number of rows in the test data.'
    # Assert that the list contains only 0s and 1s
    assert set(predictions).issubset({0, 1}), 'The list should contain only 0s and 1s.'

test_predict_potential_employee()
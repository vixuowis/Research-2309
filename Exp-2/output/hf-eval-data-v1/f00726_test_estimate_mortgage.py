def test_estimate_mortgage():
    # Load the test dataset
    test_data = pd.read_csv('jwan2021/autotrain-data-us-housing-prices')
    # Select a sample from the dataset
    sample_data = test_data.sample(n=5)
    # Call the function with the sample data
    predictions = estimate_mortgage(sample_data)
    # Assert that the function returns a list of predictions
    assert isinstance(predictions, list), 'The function should return a list of predictions.'
    # Assert that the length of the predictions matches the length of the sample data
    assert len(predictions) == len(sample_data), 'The number of predictions should match the number of samples.'
    # Assert that all predictions are numbers
    for prediction in predictions:
        assert isinstance(prediction, (int, float)), 'All predictions should be numbers.'

test_estimate_mortgage()
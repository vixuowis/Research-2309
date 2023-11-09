def test_predict_carbon_emissions():
    # Load the test data
    test_data = pd.read_csv('test_data.csv')
    # Get the predictions
    predictions = predict_carbon_emissions('test_data.csv')
    # Assert that the predictions are not None
    assert predictions is not None
    # Assert that the predictions are not empty
    assert len(predictions) > 0
    # Assert that the predictions are of the correct type
    assert isinstance(predictions, np.ndarray)
    # Assert that the predictions have the correct shape
    assert predictions.shape == (test_data.shape[0],)
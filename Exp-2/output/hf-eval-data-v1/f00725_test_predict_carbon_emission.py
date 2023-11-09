def test_predict_carbon_emission():
    # Load test dataset
    test_data = pd.read_csv('omarques/autotrain-data-in-class-test-demo')
    # Select a sample from the dataset
    sample_data = test_data.sample(n=10)
    # Save the sample data to a CSV file
    sample_data.to_csv('sample_data.csv', index=False)
    # Predict carbon emission for the sample data
    predictions = predict_carbon_emission('sample_data.csv')
    # Assert the predictions are in the expected format
    assert isinstance(predictions, np.ndarray), 'Expected a numpy array'
    assert len(predictions) == 10, 'Expected 10 predictions'
    # Assert the predictions are binary (0 or 1)
    assert set(predictions).issubset({0, 1}), 'Expected only 0s and 1s in predictions'
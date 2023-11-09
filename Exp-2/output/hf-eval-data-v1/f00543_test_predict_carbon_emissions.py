def test_predict_carbon_emissions():
    # Load test data
    test_data = pd.read_csv('test_data.csv')

    # Predict carbon emissions
    predictions = predict_carbon_emissions('test_data.csv')

    # Check if the predictions are in the expected format (numpy array)
    assert isinstance(predictions, np.ndarray), 'Expected a numpy array'

    # Check if the number of predictions matches the number of test samples
    assert len(predictions) == len(test_data), 'Number of predictions does not match number of test samples'

    # Check if the predictions are within a reasonable range (assuming carbon emissions are positive)
    assert all(i >= 0 for i in predictions), 'Negative carbon emissions predicted'

    print('All tests passed.')

test_predict_carbon_emissions()
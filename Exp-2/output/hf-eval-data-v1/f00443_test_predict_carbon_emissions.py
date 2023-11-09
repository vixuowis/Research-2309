def test_predict_carbon_emissions():
    # Load test dataset
    test_data = pd.read_csv('test_data.csv')

    # Predict carbon emissions for the test dataset
    predictions = predict_carbon_emissions(test_data)

    # Load expected results
    expected_results = pd.read_csv('expected_results.csv')

    # Compare predictions with expected results
    for prediction, expected in zip(predictions, expected_results):
        assert abs(prediction - expected) < 0.01, 'Test failed!'

    print('All tests passed!')

test_predict_carbon_emissions()
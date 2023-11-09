def test_predict_carbon_emissions():
    # Test the function with a sample dataset
    predictions = predict_carbon_emissions('sample_data.csv')
    # Load the actual values
    actual_values = pd.read_csv('sample_data.csv')['carbon_emissions']
    # Check if the predictions are close to the actual values
    assert np.allclose(predictions, actual_values, rtol=0.1), 'Test failed!'

test_predict_carbon_emissions()
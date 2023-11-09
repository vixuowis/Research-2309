def test_predict_carbon_emissions():
    # Load the test data
    test_data = pd.read_csv('test_data.csv')
    
    # Predict the carbon emissions for the test data
    predictions = predict_carbon_emissions('test_data.csv')
    
    # Load the actual carbon emissions for the test data
    actual = test_data['carbon_emissions']
    
    # Compare the predicted and actual carbon emissions
    # Note: We're using a tolerance value because exact equality is not expected
    assert np.allclose(predictions, actual, rtol=1e-05, atol=1e-08), 'Test failed!'
    
    print('Test passed!')

# Run the test function
test_predict_carbon_emissions()
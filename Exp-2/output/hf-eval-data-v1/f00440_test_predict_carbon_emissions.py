def test_predict_carbon_emissions():
    # Load the test dataset
    data = pd.read_csv('test_data.csv')
    
    # Select a sample from the dataset
    sample = data.sample()
    
    # Get the features from the sample
    feat_x1 = sample['feat_x1'].values[0]
    feat_x2 = sample['feat_x2'].values[0]
    feat_x3 = sample['feat_x3'].values[0]
    
    # Get the actual carbon emissions category
    actual = sample['carbon_emissions_category'].values[0]
    
    # Predict the carbon emissions category
    predicted = predict_carbon_emissions(feat_x1, feat_x2, feat_x3)
    
    # Check if the predicted category is close to the actual category
    assert abs(predicted - actual) < 0.1, 'Test failed!'
    
    print('Test passed!')

# Run the test function
test_predict_carbon_emissions()
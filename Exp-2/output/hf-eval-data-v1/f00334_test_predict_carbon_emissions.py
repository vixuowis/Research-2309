def test_predict_carbon_emissions():
    # Define the paths to the files
    model_file = 'model.joblib'
    config_file = 'config.json'
    data_file = 'data.csv'
    
    # Call the function with the paths to the files
    predictions = predict_carbon_emissions(model_file, config_file, data_file)
    
    # Load the actual data
    actual_data = pd.read_csv('pcoloc/autotrain-data-dragino-7-7-max_300m')
    
    # Select a sample from the actual data
    sample = actual_data.sample(n=5)
    
    # Make predictions on the sample
    sample_predictions = model.predict(sample)
    
    # Assert that the predictions are close to the actual values
    for i in range(len(sample_predictions)):
        assert abs(predictions[i] - sample_predictions[i]) < 0.1
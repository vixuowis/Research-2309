def test_predict_housing_prices():
    # Define the paths to the model, dataset, and configuration file
    model_path = 'model.joblib'
    data_path = 'data.csv'
    config_path = 'config.json'
    
    # Call the function with the defined paths
    predictions = predict_housing_prices(model_path, data_path, config_path)
    
    # Load the dataset
    data = pd.read_csv(data_path)
    
    # Assert that the number of predictions matches the number of rows in the dataset
    assert len(predictions) == len(data), 'Number of predictions does not match number of rows in dataset'
    
    # Assert that the predictions are not all the same (i.e., the model is not just predicting the mean)
    assert len(set(predictions)) > 1, 'All predictions are the same'
    
    # Assert that the predictions are not NaN
    assert not pd.isnull(predictions).any(), 'Some predictions are NaN'
    
    # Call the function to test it
    test_predict_housing_prices()
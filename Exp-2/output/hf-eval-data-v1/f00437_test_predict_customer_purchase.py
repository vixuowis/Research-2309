def test_predict_customer_purchase():
    # Define the paths to the model and data
    model_path = 'your_trained_model.joblib'
    data_path = 'customer_browsing_data.csv'

    # Call the function with the paths
    predictions = predict_customer_purchase(model_path, data_path)

    # Load the actual data
    actual_data = pd.read_csv(data_path)

    # Assert that the predictions are not null and have the same length as the actual data
    assert predictions is not None
    assert len(predictions) == len(actual_data)

    # Assert that the predictions are in the range [0, 1]
    assert all(0 <= prediction <= 1 for prediction in predictions)

test_predict_customer_purchase()
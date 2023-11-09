def test_predict_wine_quality():
    '''
    This function tests the predict_wine_quality function.
    It loads the wine quality dataset, selects several samples from the dataset, and asserts that the function returns a prediction for each sample.
    '''
    # Load the wine quality dataset
    data_file = cached_download(hf_hub_url(REPO_ID, data_filename))
    # Read the dataset using pandas
    wine_df = pd.read_csv(data_file, sep=';')
    # Select several samples from the dataset
    X_test = wine_df.drop(['quality'], axis=1).sample(5)
    Y_test = wine_df['quality'].sample(5)
    # Call the predict_wine_quality function
    labels, model_score = predict_wine_quality()
    # Assert that the function returns a prediction for each sample
    assert len(labels) == len(X_test), 'The number of predictions does not match the number of samples'
    # Assert that the model score is a float
    assert isinstance(model_score, float), 'The model score is not a float'
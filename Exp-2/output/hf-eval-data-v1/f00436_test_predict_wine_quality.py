def test_predict_wine_quality():
    # Load the dataset
    data_file = cached_download(hf_hub_url(REPO_ID, 'winequality-red.csv'))
    wine_df = pd.read_csv(data_file, sep=';')
    X = wine_df.drop(['quality'], axis=1)
    Y = wine_df['quality']
    
    # Select a few samples from the dataset
    X_sample = X.sample(n=5)
    Y_sample = Y.loc[X_sample.index]
    
    # Predict the wine quality for the samples
    Y_pred = predict_wine_quality(X_sample)
    
    # Check that the function returns the correct number of predictions
    assert len(Y_pred) == len(Y_sample), 'Number of predictions does not match number of samples'
    
    # Check that the function returns a prediction for each sample
    for pred in Y_pred:
        assert pred is not None, 'Function returned None for a prediction'
    
    # Check that the function returns a prediction in the correct range
    for pred in Y_pred:
        assert 0 <= pred <= 10, 'Function returned a prediction outside the expected range'

test_predict_wine_quality()
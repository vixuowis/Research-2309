def test_classify_wine_quality():
    """
    This function tests the classify_wine_quality function by using a sample of the winequality-red.csv dataset.
    The test is successful if the function does not throw an error and returns predictions for all samples.
    """
    # Load the test dataset
    data_file = cached_download(hf_hub_url(REPO_ID, 'winequality-red.csv'))
    winedf = pd.read_csv(data_file, sep=';')
    X = winedf.drop(['quality'], axis=1)
    # Get a sample of the dataset
    X_sample = X.sample(n=5)
    # Get predictions for the sample
    labels = classify_wine_quality(X_sample)
    # Check if the function returns the correct number of predictions
    assert len(labels) == len(X_sample), 'Number of predictions does not match number of samples'
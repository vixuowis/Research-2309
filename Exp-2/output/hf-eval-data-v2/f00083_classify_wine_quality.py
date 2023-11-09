# function_import --------------------

from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd

# function_code --------------------

def classify_wine_quality(X):
    """
    This function classifies the quality of wine based on given features.

    Args:
        X (DataFrame): A DataFrame containing the features describing the wine.

    Returns:
        labels (array): An array containing the predicted quality labels for the given wine samples.
    """
    REPO_ID = 'julien-c/wine-quality'
    FILENAME = 'sklearn_model.joblib'

    model = joblib.load(cached_download(hf_hub_url(REPO_ID, FILENAME)))

    labels = model.predict(X)
    return labels

# test_function_code --------------------

def test_classify_wine_quality():
    """
    This function tests the classify_wine_quality function by using a sample dataset.
    """
    REPO_ID = 'julien-c/wine-quality'
    data_file = cached_download(hf_hub_url(REPO_ID, 'winequality-red.csv'))
    winedf = pd.read_csv(data_file, sep=';')
    X = winedf.drop(['quality'], axis=1)
    Y = winedf['quality']

    predicted_labels = classify_wine_quality(X[:3])
    assert len(predicted_labels) == 3, 'The number of predictions is not correct.'
    assert all(isinstance(label, (int, float)) for label in predicted_labels), 'The type of predictions is not correct.'

# call_test_function_code --------------------

test_classify_wine_quality()
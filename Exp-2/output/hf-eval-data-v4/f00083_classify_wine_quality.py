# requirements_file --------------------

!pip install -U huggingface_hub joblib pandas

# function_import --------------------

from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd

# function_code --------------------

def classify_wine_quality(features):
    """
    Classify the quality of wine based on input features.

    Parameters:
    features (DataFrame): The input features describing the wine samples.

    Returns:
    np.array: An array of predicted wine quality labels.
    """
    REPO_ID = 'julien-c/wine-quality'
    FILENAME = 'sklearn_model.joblib'
    model = joblib.load(cached_download(hf_hub_url(REPO_ID, FILENAME)))
    return model.predict(features)

# test_function_code --------------------

def test_classify_wine_quality():
    print('Testing wine quality classification function')
    data_file = cached_download(hf_hub_url('julien-c/wine-quality', 'winequality-red.csv'))
    winedf = pd.read_csv(data_file, sep=';')

    X = winedf.drop(['quality'], axis=1)
    Y = winedf['quality']

    predictions = classify_wine_quality(X[:3])
    assert len(predictions) == 3, 'Expected 3 predictions, got {}'.format(len(predictions))
    assert all(isinstance(label, int) for label in predictions), 'Predictions should be integers'
    print('Test passed!')
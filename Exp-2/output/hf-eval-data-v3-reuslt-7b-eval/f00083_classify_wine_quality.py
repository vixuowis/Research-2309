# function_import --------------------

from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd
import numpy as np

# function_code --------------------

def classify_wine_quality(X):
    '''
    Classify the quality of wine based on given features.
    
    Args:
        X (pandas.DataFrame): The features of the wine samples.
    
    Returns:
        numpy.ndarray: The predicted labels of the wine samples.
    
    Raises:
        ValueError: If the input is not a pandas DataFrame.
    '''
    
    # Load the model from Hub.
    url = hf_hub_url('c0nn3x/wine-quality', 'model.joblib')
    local_path = cached_download(url)
    model = joblib.load(local_path)
    
    if isinstance(X, pd.DataFrame):
        Yhat = model.predict(X)
        return Yhat
    else:
        raise ValueError("Input X must be of type pandas DataFrame.")



# test_function_code --------------------

def test_classify_wine_quality():
    '''
    Test the function classify_wine_quality.
    '''
    data_file = cached_download(hf_hub_url('julien-c/wine-quality', 'winequality-red.csv'))
    winedf = pd.read_csv(data_file, sep=';')
    
    X = winedf.drop(['quality'], axis=1)
    
    predicted_labels = classify_wine_quality(X[:3])
    
    assert isinstance(predicted_labels, np.ndarray), 'The type of predictions is not correct.'
    assert len(predicted_labels) == 3, 'The number of predictions is not correct.'
    
    return 'All Tests Passed'


# call_test_function_code --------------------

test_classify_wine_quality()
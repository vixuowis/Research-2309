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
    
    if not isinstance(X, pd.DataFrame): raise ValueError('Input data must be in a pandas DataFrame')
    
    # load model
    model_url = hf_hub_url("datalens-ai", "function_import/function_code/model")
    model_path = cached_download(model_url)
    model = joblib.load(model_path) 
    
    # predict
    y = model.predict(X)
    
    return np.where(y > 5, 'good', 'bad')

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
# function_import --------------------

from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd
import numpy as np

# function_code --------------------

def predict_wine_quality():
    '''
    This function loads a Scikit-learn model from the Hugging Face Hub and uses it to predict wine quality.
    
    Returns:
        labels (numpy.ndarray): The predicted quality labels for the wines.
    '''
    REPO_ID = 'julien-c/wine-quality'
    FILENAME = 'sklearn_model.joblib'
    
    model = joblib.load(cached_download(hf_hub_url(REPO_ID, FILENAME)))
    data_file = cached_download(hf_hub_url(REPO_ID, 'winequality-red.csv'))
    wine_df = pd.read_csv(data_file, sep=';')
    X = wine_df.drop(['quality'], axis=1)
    
    labels = model.predict(X)
    return labels

# test_function_code --------------------

def test_predict_wine_quality():
    '''
    This function tests the predict_wine_quality function by checking the shape and dtype of the output.
    '''
    labels = predict_wine_quality()
    assert isinstance(labels, np.ndarray), 'Output should be a numpy array.'
    assert labels.shape[0] == 1599, 'Output shape should be (1599,).'
    assert labels.dtype == np.int64, 'Output dtype should be int64.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_predict_wine_quality()
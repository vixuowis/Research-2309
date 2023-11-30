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
    
    # Load pre-trained scikitlearn model from huggingface hub if it has not already been downloaded
    model_path = 'marecchi/wine-quality'
    model_file = 'model.joblib'
    cached_download(hf_hub_url(repo_id=model_path, filename=model_file))
    
    # Load the scikitlearn model
    clf = joblib.load('./model.joblib') 

    return clf

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
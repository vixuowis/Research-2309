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
    # Load model
    url = hf_hub_url('GuillaumeDuv/test-model-skip-CI', 'master')
    path = cached_download(url=url)
    model = joblib.load(path)
    
    # Load data (already in the HF hub)
    df_wine = pd.read_csv('./data/processed/test-model-skip-CI/dataset/dataset.csv')
    X = np.array([list(df_wine['fixed acidity']), list(df_wine['volatile acidity'])])
    
    # Make prediction
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
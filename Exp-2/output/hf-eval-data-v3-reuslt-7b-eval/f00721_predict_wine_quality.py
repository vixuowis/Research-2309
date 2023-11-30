# function_import --------------------

from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd
import numpy as np

# function_code --------------------

def predict_wine_quality():
    '''
    This function is used to predict the quality of wine based on its chemical properties.
    It uses a pre-trained model hosted on Hugging Face hub.
    
    Returns:
        tuple: A tuple containing the predicted labels and the model's score.
    
    Raises:
        Exception: If there is an error in loading the model or the data.
    '''
    try:
        # Load the dataset
        url = hf_hub_url('shivam1402/ml-model', "winequality.csv", revision= 'd73e28b695a4566c6725789b681828960223f176')
        df = pd.read_csv(cached_download(url))
        
        # Preprocess the data and get the features
        X, y, cols = preprocessing(df)
    
        # Load the model
        url = hf_hub_url('shivam1402/ml-model', "winequality.joblib", revision= 'd73e28b695a4566c6725789b681828960223f176')
        clf = joblib.load(cached_download(url))
        
        # Predict the labels
        pred_y, score = clf.predict(X), clf.score(X, y) * 100
        
        return (pred_y, round(np.mean(score)))
    
    except Exception as e:
        print("Exception occurred in predicting the quality of wine.")
        raise
    
# function_code --------------------


# test_function_code --------------------

def test_predict_wine_quality():
    '''
    This function is used to test the predict_wine_quality function.
    It checks if the function returns the correct output type and if the model score is within an acceptable range.
    '''
    labels, score = predict_wine_quality()
    assert isinstance(labels, np.ndarray), 'The predicted labels should be a numpy array.'
    assert isinstance(score, float), 'The model score should be a float.'
    assert 0 <= score <= 1, 'The model score should be between 0 and 1.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_predict_wine_quality()
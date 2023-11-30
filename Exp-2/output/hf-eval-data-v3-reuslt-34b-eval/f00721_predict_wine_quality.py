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
        
        # Model download link from hugging face hub
        WINE_QUALITY_MODEL = 'https://huggingface.co/aimeurussink/wine-quality'
        
        # Download the model and feature names using hf_hub_download()
        try:
            artifact_url = hf_hub_url(repo_id=WINE_QUALITY_MODEL, 
                                      filename='preprocessor.joblib')
            
            # Download preprocessing file
            cached_download(artifact_url, cache_dir=".", 
                            library_name="model", library_version="v1")
        except Exception as e:
            return None, str(e)
        
        try:    
            artifact_url = hf_hub_url(repo_id=WINE_QUALITY_MODEL, 
                                      filename='model.joblib')
            
            # Download model file
            cached_download(artifact_url, cache_dir=".", 
                            library_name="model", library_version="v1")
        except Exception as e:
            return None, str(e)
        
        # Load the preprocessor and model files
        try:
            processor = joblib.load('preprocessor.joblib')
            
            model = joblib.load('model.joblib')
        except Exception as e:
            return None, str(e)
    
    except Exception as e:
        # If there is an error in reading the files, then return None
        return None, str(e)
        
    try:
        X = pd.read_json(req["X"])
    except Exception as e:
        # If there is an error in loading the data, then return None
        return None, str(e)
    
    try:
        X_processed = processor.transform(X)
        
        preds = model.predict(X_processed)
    except Exception as e:
        # If there is an error in making predictions, then return None
        return None, str(e)
    
    try:


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
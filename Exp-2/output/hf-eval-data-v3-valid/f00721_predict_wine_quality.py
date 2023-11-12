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
    REPO_ID = 'julien-c/wine-quality'
    FILENAME = 'sklearn_model.joblib'
    data_filename = 'winequality-red.csv'

    try:
        model = joblib.load(cached_download(hf_hub_url(REPO_ID, FILENAME)))
        data_file = cached_download(hf_hub_url(REPO_ID, data_filename))
    except Exception as e:
        raise Exception('Error in loading model or data: ' + str(e))

    wine_df = pd.read_csv(data_file, sep=';')
    X = wine_df.drop(['quality'], axis=1)
    Y = wine_df['quality']

    labels = model.predict(X)
    model_score = model.score(X, Y)

    return labels, model_score

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
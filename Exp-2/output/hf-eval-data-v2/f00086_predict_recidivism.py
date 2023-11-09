# function_import --------------------

from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd
import numpy as np

# function_code --------------------

def predict_recidivism(X_test):
    '''
    Predicts future criminal re-offense using a pre-trained model.
    
    Args:
        X_test (DataFrame): The test dataset for which to predict recidivism.
    
    Returns:
        predictions (array): The predicted labels for the test dataset.
    '''
    REPO_ID = 'imodels/figs-compas-recidivism'
    FILENAME = 'sklearn_model.joblib'
    model = joblib.load(cached_download(hf_hub_url(REPO_ID, FILENAME)))
    predictions = model.predict(X_test)
    return predictions

# test_function_code --------------------

def test_predict_recidivism():
    '''
    Tests the predict_recidivism function by comparing the predicted labels with the ground truth labels.
    
    Raises:
        AssertionError: If the accuracy of the predictions is not within an acceptable range.
    '''
    X_test = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
    y_test = np.random.randint(0,2,size=(100,))
    predictions = predict_recidivism(X_test)
    accuracy = np.mean(predictions == y_test)
    assert 0.6 <= accuracy <= 0.7, f'Expected accuracy between 0.6 and 0.7, but got {accuracy}'

# call_test_function_code --------------------

test_predict_recidivism()
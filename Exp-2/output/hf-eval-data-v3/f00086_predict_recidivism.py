# function_import --------------------

from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd
import numpy as np

# function_code --------------------

def predict_recidivism(X_test):
    """
    This function loads a pre-trained model from Hugging Face Hub and uses it to predict recidivism.
    
    Args:
        X_test (pd.DataFrame): The test data to be used for prediction.
    
    Returns:
        np.array: The predicted labels for the test data.
    
    Raises:
        ModuleNotFoundError: If the required modules are not installed.
    """
    REPO_ID = 'imodels/figs-compas-recidivism'
    FILENAME = 'sklearn_model.joblib'
    model = joblib.load(cached_download(hf_hub_url(REPO_ID, FILENAME)))
    predictions = model.predict(X_test)
    return predictions

# test_function_code --------------------

def test_predict_recidivism():
    """
    This function tests the predict_recidivism function.
    """
    X_test = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
    predictions = predict_recidivism(X_test)
    assert isinstance(predictions, np.ndarray), 'The prediction result should be a numpy array.'
    assert len(predictions) == len(X_test), 'The number of predictions should match the number of test instances.'
    print('All Tests Passed')

# call_test_function_code --------------------

test_predict_recidivism()
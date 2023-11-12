# function_import --------------------

from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# function_code --------------------

def predict_vacation_success(destination: str, accommodation: str, travel_style: str) -> int:
    '''
    Predicts whether a client's vacation will be successful based on their chosen destination, accommodation, and travel style.
    
    Args:
        destination (str): The chosen destination of the client.
        accommodation (str): The chosen accommodation of the client.
        travel_style (str): The chosen travel style of the client.
    
    Returns:
        int: 1 if the vacation is predicted to be successful, 0 otherwise.
    '''
    REPO_ID = 'danupurnomo/dummy-titanic'
    PIPELINE_FILENAME = 'final_pipeline.pkl'
    TF_FILENAME = 'titanic_model.h5'

    model_pipeline = joblib.load(cached_download(hf_hub_url(REPO_ID, PIPELINE_FILENAME)))
    model_seq = load_model(cached_download(hf_hub_url(REPO_ID, TF_FILENAME)))

    new_data = pd.DataFrame({'destination': [destination], 'accommodation': [accommodation], 'travel_style': [travel_style]})
    prediction = model_seq.predict(model_pipeline.transform(new_data))
    success = (prediction > 0.5).astype(int)
    return success[0]

# test_function_code --------------------

def test_predict_vacation_success():
    assert predict_vacation_success('Bali', 'Hotel', 'Solo') == 1
    assert predict_vacation_success('Paris', 'Hostel', 'Group') == 0
    assert predict_vacation_success('New York', 'Apartment', 'Family') == 1
    return 'All Tests Passed'

# call_test_function_code --------------------

test_predict_vacation_success()
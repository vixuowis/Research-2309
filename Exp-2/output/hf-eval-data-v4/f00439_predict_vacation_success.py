# requirements_file --------------------

!pip install -U huggingface_hub joblib pandas numpy tensorflow

# function_import --------------------

from huggingface_hub import hf_hub_url, cached_download
from tensorflow.keras.models import load_model
import joblib
import pandas as pd

# function_code --------------------

def predict_vacation_success(destination, accommodation, travel_style):
    """
    Predict the success of a client's vacation based on destination, accommodation, and travel style.

    Parameters:
    - destination (str): The vacation destination.
    - accommodation (str): Type of accommodation.
    - travel_style (str): The client's travel style.

    Returns:
    - int: 1 if vacation success is likely, 0 otherwise.
    """
    REPO_ID = 'danupurnomo/dummy-titanic'
    PIPELINE_FILENAME = 'final_pipeline.pkl'
    TF_FILENAME = 'titanic_model.h5'

    model_pipeline = joblib.load(cached_download(hf_hub_url(REPO_ID, PIPELINE_FILENAME)))
    model_seq = load_model(cached_download(hf_hub_url(REPO_ID, TF_FILENAME)))

    new_data = pd.DataFrame({
        "destination": [destination], 
        "accommodation": [accommodation], 
        "travel_style": [travel_style]
    })

    prediction = model_seq.predict(model_pipeline.transform(new_data))
    success = (prediction > 0.5).astype(int)
    return success.item()

# test_function_code --------------------

def test_predict_vacation_success():
    print("Testing predict_vacation_success function.")

    # Test case 1: Expected success
    assert predict_vacation_success('Bali', 'Hotel', 'Solo') == 1, "Test case 1 failed: Expected successful prediction."

    # Test case 2: Expected failure
    assert predict_vacation_success('Antarctica', 'Tent', 'Adventure') == 0, "Test case 2 failed: Expected unsuccessful prediction."

    # Test case 3: Test edge case
    assert predict_vacation_success('', '', '') == 0, "Test case 3 failed: Expected unsuccessful prediction for missing data."
    print("All tests passed.")
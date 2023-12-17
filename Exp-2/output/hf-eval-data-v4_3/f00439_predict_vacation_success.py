# requirements_file --------------------

import subprocess

requirements = ["huggingface_hub", "joblib", "pandas", "tensorflow"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd
from tensorflow.keras.models import load_model

# function_code --------------------

def predict_vacation_success(destination, accommodation, travel_style):
    """
    Predict the success of the client's vacation based on destination, accommodation, and travel style.

    Args:
        destination (str): The destination of the vacation.
        accommodation (str): The type of accommodation.
        travel_style (str): The style of travel.

    Returns:
        int: 1 if the vacation is predicted to be successful, 0 otherwise.

    Raises:
        ValueError: If any of the inputs are not a string type.
    """
    if not all(isinstance(arg, str) for arg in [destination, accommodation, travel_style]):
        raise ValueError('All arguments must be of type str.')

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
    return (prediction > 0.5).astype(int)[0]

# test_function_code --------------------

def test_predict_vacation_success():
    print("Testing started.")
    # Test case 1: Expected success
    print("Testing case [1/3] started.")
    assert predict_vacation_success('Bali', 'Resort', 'Family') == 1, f"Test case [1/3] failed: Predicted failure for a typically successful vacation setup."

    # Test case 2: Expected failure
    print("Testing case [2/3] started.")
    assert predict_vacation_success('Siberia', 'Tent', 'Solo') == 0, f"Test case [2/3] failed: Predicted success for a typically challenging vacation setup."

    # Test case 3: Invalid inputs
    try:
        print("Testing case [3/3] started.")
        predict_vacation_success(123, 'Hotel', 'Solo')
        raise AssertionError('Test case [3/3] failed: ValueError was not raised for invalid input types.')
    except ValueError:
        pass # Passed the test, ValueError was raised

    print("Testing finished.")

# call_test_function_line --------------------

test_predict_vacation_success()
from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd
import numpy as np

# Function to predict recidivism
# This function loads a pre-trained model from Hugging Face Hub and uses it to predict recidivism
# The model is trained on the COMPAS dataset
# The function takes as input a pandas DataFrame (X_test) and returns the predicted labels

def predict_recidivism(X_test):
    REPO_ID = 'imodels/figs-compas-recidivism'
    FILENAME = 'sklearn_model.joblib'
    model = joblib.load(cached_download(hf_hub_url(REPO_ID, FILENAME)))
    predictions = model.predict(X_test)
    return predictions
from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd

REPO_ID = 'julien-c/wine-quality'
FILENAME = 'sklearn_model.joblib'

# Load the Scikit-learn model
model = joblib.load(cached_download(hf_hub_url(REPO_ID, FILENAME)))

def predict_wine_quality(X):
    '''
    This function takes in a pandas dataframe of wine features and returns a prediction of the wine quality.
    '''
    # Predict wine quality
    labels = model.predict(X)
    return labels
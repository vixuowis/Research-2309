from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd

REPO_ID = 'julien-c/wine-quality'
FILENAME = 'sklearn_model.joblib'

# Function to classify wine quality
def classify_wine_quality(X):
    """
    This function takes in a dataframe of wine features and returns the predicted quality labels.
    The model used for prediction is a pre-trained sklearn model hosted on HuggingFace.
    """
    # Load the pre-trained model
    model = joblib.load(cached_download(hf_hub_url(REPO_ID, FILENAME)))
    # Get predictions for the given features
    labels = model.predict(X)
    return labels
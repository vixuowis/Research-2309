from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd

REPO_ID = 'julien-c/wine-quality'
FILENAME = 'sklearn_model.joblib'
data_filename = 'winequality-red.csv'

# Function to predict wine quality
def predict_wine_quality():
    '''
    This function loads a pre-trained model and a wine quality dataset from Hugging Face hub.
    It then uses the model to predict the wine quality (good or bad) based on the given chemical properties of the wine samples.
    '''
    # Load the pre-trained model
    model = joblib.load(cached_download(hf_hub_url(REPO_ID, FILENAME)))
    # Load the wine quality dataset
    data_file = cached_download(hf_hub_url(REPO_ID, data_filename))
    # Read the dataset using pandas
    wine_df = pd.read_csv(data_file, sep=';')
    # Separate the input features (X) and the target variable (Y)
    X = wine_df.drop(['quality'], axis=1)
    Y = wine_df['quality']
    # Use the pre-trained model to predict the wine quality
    labels = model.predict(X)
    # Evaluate the accuracy of the model on the dataset
    model_score = model.score(X, Y)
    return labels, model_score
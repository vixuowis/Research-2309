import joblib
import pandas as pd
import json

# Function to predict the species of plants among Iris Setosa, Iris Versicolor, and Iris Virginica
# using a pre-trained K-Nearest Neighbors (KNN) model.
def predict_plant_species(data):
    '''
    This function takes a pandas DataFrame as input, uses a pre-trained KNN model to predict the species of plants.
    The input DataFrame should have the same feature columns as the Iris dataset.
    '''
    # Load the pre-trained KNN model
    model = joblib.load('model.joblib')
    # Load the config.json file to retrieve the input features required for the model
    config = json.load(open('config.json'))
    features = config['features']
    # Extract only the required features from the input DataFrame
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]
    # Perform predictions using the pre-trained KNN model
    predictions = model.predict(data)
    return predictions
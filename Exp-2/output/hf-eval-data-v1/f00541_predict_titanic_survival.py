import joblib
import pandas as pd
import json

# Function to predict the survival of passengers on the Titanic
# The function takes a CSV file as input and returns the predicted survival probabilities
# The model used for prediction is loaded from a pretrained model on Hugging Face

def predict_titanic_survival(data_file):
    # Load the pretrained model
    model = joblib.load('model.joblib')
    # Load the model configuration
    config = json.load(open('config.json'))
    features = config['features']
    # Read the data from the CSV file
    data = pd.read_csv(data_file)
    # Select the relevant features from the data
    data = data[features]
    # Rename the columns according to the feature names in the model's config file
    data.columns = ['feat_' + str(col) for col in data.columns]
    # Predict the survival probabilities
    predictions = model.predict(data)
    return predictions
import json
import joblib
import pandas as pd

# Function to predict carbon emissions
# This function loads a pre-trained regression model and uses it to predict carbon emissions for a new line of electric vehicles
# The function takes as input the path to the new vehicle data and returns the predicted carbon emissions

def predict_carbon_emissions(data_path):
    # Load the pre-trained model
    model = joblib.load('model.joblib')
    # Load the configuration file
    config = json.load(open('config.json'))
    # Get the features from the configuration file
    features = config['features']
    # Load the new vehicle data
    data = pd.read_csv(data_path)
    # Select the necessary features
    data = data[features]
    # Rename the columns
    data.columns = ['feat_' + str(col) for col in data.columns]
    # Make predictions
    predictions = model.predict(data)
    # Return the predictions
    return predictions
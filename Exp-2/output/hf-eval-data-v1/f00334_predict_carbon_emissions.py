import json
import joblib
import pandas as pd

# Function to predict carbon emissions
# Parameters: 
# model_file: string, path to the trained model file
# config_file: string, path to the configuration file
# data_file: string, path to the new data file
# Returns: array, predictions of carbon emissions

def predict_carbon_emissions(model_file, config_file, data_file):
    # Load the pre-trained model
    model = joblib.load(model_file)
    
    # Load the configuration file
    config = json.load(open(config_file))
    
    # Get the features from the configuration file
    features = config['features']
    
    # Load the new data
    data = pd.read_csv(data_file)
    
    # Pre-process the data
    data = data[features]
    data.columns = ['feat_' + str(col) for col in data.columns]
    
    # Make predictions
    predictions = model.predict(data)
    
    return predictions
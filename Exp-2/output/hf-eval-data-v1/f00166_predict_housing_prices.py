import joblib
import pandas as pd
import json

# Function to predict housing prices
# Parameters:
# - model_path: Path to the trained model
# - data_path: Path to the dataset
# - config_path: Path to the configuration file
# Returns:
# - Predicted housing prices

def predict_housing_prices(model_path, data_path, config_path):
    # Load the model
    model = joblib.load(model_path)
    
    # Load the dataset
    data = pd.read_csv(data_path)
    
    # Load the configuration file
    config = json.load(open(config_path))
    
    # Filter the dataset to only include the columns specified in the 'features' list
    features = config['features']
    data = data[features]
    
    # Rename the columns using 'feat_' prefix
    data.columns = ['feat_' + str(col) for col in data.columns]
    
    # Use the pre-trained model to predict the housing prices
    predictions = model.predict(data)
    
    return predictions
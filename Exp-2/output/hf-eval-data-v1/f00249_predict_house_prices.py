import joblib
import pandas as pd
import json

# Function to predict US house prices
# The function loads a pre-trained model and a configuration file
# It then loads a dataset and selects the relevant features
# Finally, it uses the model to predict the house prices

def predict_house_prices(data_file):
    # Load the pre-trained model
    model = joblib.load('model.joblib')
    
    # Load the configuration file
    config = json.load(open('config.json'))
    features = config['features']
    
    # Load the dataset
    data = pd.read_csv(data_file)
    
    # Select the relevant features
    data = data[features]
    
    # Rename the columns as required by the pre-trained model
    data.columns = ['feat_' + str(col) for col in data.columns]
    
    # Use the model to predict the house prices
    predictions = model.predict(data)
    
    return predictions
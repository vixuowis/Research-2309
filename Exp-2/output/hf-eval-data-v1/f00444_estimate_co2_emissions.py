import joblib
import pandas as pd
import json

# Function to estimate CO2 emissions based on historic data
# The function loads a trained model, reads the client's historic data from a CSV file,
# selects the required features, renames the columns to match the format required by the model,
# and uses the predict() method of the loaded model to output the estimated CO2 emissions.
def estimate_co2_emissions(data_file):
    # Load the trained model
    model = joblib.load('model.joblib')
    # Load the configuration file
    config = json.load(open('config.json'))
    # Get the features from the configuration file
    features = config['features']
    # Read the client's historic data from the CSV file
    data = pd.read_csv(data_file)
    # Select the required features
    data = data[features]
    # Rename the columns to match the format required by the model
    data.columns = ['feat_' + str(col) for col in data.columns]
    # Use the predict() method of the loaded model to output the estimated CO2 emissions
    predictions = model.predict(data)
    return predictions
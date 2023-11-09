import json
import joblib
import pandas as pd

# Function to predict whether a chemical plant is exceeding carbon emission limits
# based on a CSV file containing data collected.
def predict_carbon_emission(data_file):
    # Load the classifier model
    model = joblib.load('model.joblib')
    # Load the configuration file containing the features used in the model
    config = json.load(open('config.json'))
    features = config['features']
    # Read the data in the CSV file
    data = pd.read_csv(data_file)
    # Select only the specified features columns
    data = data[features]
    # Rename the columns according to the model's expectation
    data.columns = ['feat_' + str(col) for col in data.columns]
    # Use the loaded model to predict whether the plant is exceeding carbon emission limits
    predictions = model.predict(data)
    return predictions
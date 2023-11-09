import joblib
import pandas as pd

# Function to estimate mortgage for a given housing using the housing's features
# The function loads a pre-trained model for US housing prices prediction
# It then uses this model to predict the mortgage based on the given housing features

def estimate_mortgage(data):
    model = joblib.load('model.joblib')
    # Filter the data to only include the required features
    filtered_columns = config['features']
    data = data[filtered_columns]
    # Adjust the column names in the data to match the format the model expects
    data.columns = [f'feat_{col}' for col in data.columns]
    # Use the model's predict function to generate mortgage estimates
    predictions = model.predict(data)
    return predictions
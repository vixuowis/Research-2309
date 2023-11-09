import joblib
import pandas as pd

# Function to estimate house price based on its features
# Input: A dictionary with feature names as keys and corresponding values
# Output: Estimated house price

def estimate_house_price(features):
    # Load the trained model
    model = joblib.load('model.joblib')

    # Prepare a dataframe with house features
    house_data = pd.DataFrame(features)

    # Make predictions
    house_price = model.predict(house_data)

    return house_price
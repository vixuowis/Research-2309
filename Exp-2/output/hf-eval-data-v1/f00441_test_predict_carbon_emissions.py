import pandas as pd
import numpy as np

# Function to test the predict_carbon_emissions function
# Input: None
# Output: None

def test_predict_carbon_emissions():
    # Define the path to the test data
    test_data_path = 'test_data.csv'
    # Load the test data
    test_data = pd.read_csv(test_data_path)
    # Get the actual carbon emissions
    actual_emissions = test_data['emissions']
    # Predict the carbon emissions
    predicted_emissions = predict_carbon_emissions(test_data_path)
    # Check if the predicted emissions are close to the actual emissions
    assert np.isclose(predicted_emissions, actual_emissions, rtol=0.1).all(), 'Test failed!'

# Run the test function
test_predict_carbon_emissions()
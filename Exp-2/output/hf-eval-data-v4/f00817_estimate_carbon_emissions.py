# requirements_file --------------------

!pip install -U json joblib pandas

# function_import --------------------

import json
import joblib
import pandas as pd

# function_code --------------------

def estimate_carbon_emissions(idle_power, standby_power, active_power):
    """
    Estimate the carbon emissions of a device based on its idle power, standby power, and active power.
    
    Parameters:
    - idle_power (float): The idle power consumption of the device in watts.
    - standby_power (float): The standby power consumption of the device in watts.
    - active_power (float): The active power consumption of the device in watts.
    
    Returns:
    - float: The estimated carbon emissions in grams.
    """
    # Load the pre-trained model
    model = joblib.load('model.joblib')
    
    # Load configuration
    config = json.load(open('config.json'))
    features = config['features']
    
    # Create a DataFrame with the incoming features
    data = pd.DataFrame([[idle_power, standby_power, active_power]], columns=features)
    
    # Rename the columns to match the expected model input
    data.columns = ['feat_' + str(col) for col in data.columns]
    
    # Predict the carbon emissions
    emissions_estimate = model.predict(data)
    
    # Return the estimated emissions
    return emissions_estimate[0]

# test_function_code --------------------

def test_estimate_carbon_emissions():
    print("Testing estimate_carbon_emissions function.")

    # Test case 1: Known values
    print("Testing case [1/3] started.")
    estimated_emissions = estimate_carbon_emissions(10, 5, 100)
    assert isinstance(estimated_emissions, float), f"Test case [1/3] failed: Expected float, got {type(estimated_emissions)}"

    # Test case 2: Edge case with zeros
    print("Testing case [2/3] started.")
    estimated_emissions = estimate_carbon_emissions(0, 0, 0)
    assert isinstance(estimated_emissions, float), f"Test case [2/3] failed: Expected float, got {type(estimated_emissions)}"

    # Test case 3: Negative inputs
    print("Testing case [3/3] started.")
    estimated_emissions = estimate_carbon_emissions(-5, -3, -20)
    assert isinstance(estimated_emissions, float), f"Test case [3/3] failed: Expected float, got {type(estimated_emissions)}"
    print("Testing finished.")

# Run the test function
test_estimate_carbon_emissions()
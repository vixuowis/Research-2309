# requirements_file --------------------

!pip install -U joblib pandas

# function_import --------------------

import json
import joblib
import pandas as pd

# function_code --------------------

def predict_carbon_emissions(features):
    """
    Predict the carbon emissions category for buildings based on their features.

    :param features: A list of feature values for the building.
    :return: The predicted carbon emissions category.
    """
    # Load the configuration for feature names
    config = json.load(open('config.json'))
    feature_names = config['features']

    # Load the model
    model = joblib.load('model.joblib')

    # Create DataFrame with the input features
    input_data = pd.DataFrame([features], columns=feature_names)

    # Make the prediction
    predictions = model.predict(input_data)

    return predictions[0]

# test_function_code --------------------

def test_predict_carbon_emissions():
    print("Testing predict_carbon_emissions function.")
    # Example features for a building
    sample_features = [100, 5, 30] # Replace with real feature values

    # Expected output format (example)
    expected_category = 'Low'

    # Test the prediction function
    predicted_category = predict_carbon_emissions(sample_features)
    assert predicted_category == expected_category, f"Test failed: expected {expected_category}, got {predicted_category}"
    print("Test passed.")

# Run the test
try:
    test_predict_carbon_emissions()
except AssertionError as e:
    print(e)
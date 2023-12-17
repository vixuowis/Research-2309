# requirements_file --------------------

!pip install -U pandas, dabl

# function_import --------------------

import pandas as pd
import dabl
from dabl import SimpleRegressor

# function_code --------------------

def predict_tip_amount(data):
    """
    Predict the tip amount for a given dataset.

    Args:
        data (pd.DataFrame): The input data containing features to predict tips.

    Returns:
        pd.Series: The predicted tip amounts.
    """
    target_column = 'tip'
    # Initialize the regressor
    regressor = SimpleRegressor()
    # Fit the model on the provided data
    model = regressor.fit(data, target=target_column)
    # Predict the tip amounts
    predicted_tips = model.predict(data)
    return predicted_tips

# test_function_code --------------------

def test_predict_tip_amount():
    print("Testing predict_tip_amount function.")
    # Create a sample dataset
    sample_data = pd.DataFrame({
        'total_bill': [34.81],
        'size': [3]
    })
    # Expected tip amount (for testing purposes, could be an approximate value)
    expected_tip_amount = 5.54 # This value would be known from a baseline or historical data
    # Call the function to predict tips
    predicted_tip = predict_tip_amount(sample_data)
    # Test if the predicted tip is close to the expected (this is a dummy value)
    assert predicted_tip.iloc[0] close to expected_tip_amount, "The predicted tip amount is not as expected."
    print("Testing completed successfully.")

# Run test
test_predict_tip_amount()
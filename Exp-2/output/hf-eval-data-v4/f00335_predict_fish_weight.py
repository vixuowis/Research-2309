# requirements_file --------------------

!pip install -U skops

# function_import --------------------

from skops.hub_utils import download
from skops.io import load

# function_code --------------------

def predict_fish_weight(fish_measurements):
    """
    Predict the weight of a fish given its measurements using a pre-trained GradientBoostingRegressor model.

    Parameters:
    fish_measurements (array-like): The measurements of the fish to predict its weight.

    Returns:
    float: The predicted weight of the fish.
    """
    # Download and load the pre-trained model
    download('brendenc/Fish-Weight', 'path_to_folder')
    model = load('path_to_folder/example.pkl')

    # Predict and return the fish weight
    predicted_weight = model.predict([fish_measurements])
    return predicted_weight[0]

# test_function_code --------------------

def test_predict_fish_weight():
    print("Testing the predict_fish_weight function.")

    # Test case 1: Measurements for a known fish sample
    test_measurements_1 = [24.0, 26.3, 31.2, 12.48, 4.3056]  # Example measurements
    expected_weight_1 = 300  # Example known weight
    predicted_weight_1 = predict_fish_weight(test_measurements_1)
    assert abs(predicted_weight_1 - expected_weight_1) < 20, f"Test case failed: Expected {expected_weight_1}, got {predicted_weight_1}"

    # Test case 2: Another known fish sample
    test_measurements_2 = [29.0, 31.2, 34.5, 16.44, 5.1373]  # Example measurements
    expected_weight_2 = 450  # Example known weight
    predicted_weight_2 = predict_fish_weight(test_measurements_2)
    assert abs(predicted_weight_2 - expected_weight_2) < 20, f"Test case failed: Expected {expected_weight_2}, got {predicted_weight_2}"

    print("All test cases passed for predict_fish_weight.")

# Run the test function
test_predict_fish_weight()
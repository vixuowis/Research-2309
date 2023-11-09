# function_import --------------------

from skops.hub_utils import download
from skops.io import load

# function_code --------------------

def predict_fish_weight(fish_measurements):
    """
    This function predicts the weight of a fish based on its measurements using a pre-trained GradientBoostingRegressor model.

    Args:
        fish_measurements (array-like): The measurements of the fish. Expected to have features compatible with the pre-trained model.

    Returns:
        predicted_weight (array-like): The predicted weight of the fish.

    Raises:
        ValueError: If the input is not array-like or if it's not compatible with the model.
    """
    # Download the model
    download('brendenc/Fish-Weight', 'path_to_folder')
    # Load the model
    model = load('path_to_folder/example.pkl')
    # Predict the weight of the fish
    predicted_weight = model.predict(fish_measurements)
    return predicted_weight

# test_function_code --------------------

def test_predict_fish_weight():
    """
    This function tests the predict_fish_weight function by using a sample fish measurement.
    """
    # A sample fish measurement
    fish_measurements = [25.4, 11.52, 4.02]
    # Call the function with the sample measurement
    predicted_weight = predict_fish_weight(fish_measurements)
    # Assert the type of the output
    assert isinstance(predicted_weight, (np.ndarray, list)), 'The output should be array-like'
    # Assert the size of the output
    assert len(predicted_weight) == len(fish_measurements), 'The output size should match the input size'

# call_test_function_code --------------------

test_predict_fish_weight()
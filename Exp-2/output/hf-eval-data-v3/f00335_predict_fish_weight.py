# function_import --------------------

from skops.hub_utils import download
from skops.io import load
from sklearn.ensemble import GradientBoostingRegressor

# function_code --------------------

def predict_fish_weight(model_path: str, fish_measurements: list) -> float:
    """
    Predicts the weight of a fish based on its measurements using a pre-trained GradientBoostingRegressor model.

    Args:
        model_path (str): The path to the pre-trained model.
        fish_measurements (list): A list of measurements of the fish.

    Returns:
        float: The predicted weight of the fish.

    Raises:
        FileNotFoundError: If the model file does not exist at the provided path.
        ValueError: If the fish_measurements list is empty or contains invalid values.
    """
    download('brendenc/Fish-Weight', model_path)
    model = load(model_path)
    predicted_weight = model.predict(fish_measurements)
    return predicted_weight

# test_function_code --------------------

def test_predict_fish_weight():
    """Tests the predict_fish_weight function."""
    model_path = 'path_to_folder/example.pkl'
    fish_measurements = [[44.0, 28.5, 30.4, 12.4]]
    assert isinstance(predict_fish_weight(model_path, fish_measurements), float), 'The function should return a float.'
    try:
        predict_fish_weight('non_existent_path', fish_measurements)
    except FileNotFoundError:
        pass
    else:
        assert False, 'The function should raise a FileNotFoundError if the model file does not exist.'
    try:
        predict_fish_weight(model_path, [])
    except ValueError:
        pass
    else:
        assert False, 'The function should raise a ValueError if the fish_measurements list is empty.'
    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_predict_fish_weight())
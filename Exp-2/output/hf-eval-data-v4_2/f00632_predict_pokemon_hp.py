# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def predict_pokemon_hp(input_data):
    """
    Predict the HP of a Pokemon using a pre-trained regression model.

    Args:
        input_data (dict): A dictionary containing the attributes of the Pokemon.

    Returns:
        float: The predicted HP value for the Pokemon.

    Raises:
        ValueError: If the input_data is not a dictionary.
    """
    if not isinstance(input_data, dict):
        raise ValueError('Input data must be a dictionary.')
    
    hp_predictor = pipeline('regression', model='julien-c/pokemon-predict-hp')
    predicted_hp = hp_predictor(input_data)[0]['score']
    return predicted_hp

# test_function_code --------------------

def test_predict_pokemon_hp():
    print("Testing started.")
    # Mock dataset or attributes for testing
    test_data = [
        {'attribute_1': 50, 'attribute_2': 100, 'attribute_3': 70},  # Expected HP around 60
        {'attribute_1': 80, 'attribute_2': 70, 'attribute_3': 40},   # Expected HP around 65
        {'attribute_1': 90, 'attribute_2': 120, 'attribute_3': 80}   # Expected HP around 100
    ]

    # Test case 1
    print("Testing case [1/3] started.")
    predicted_hp = predict_pokemon_hp(test_data[0])
    assert 55 <= predicted_hp <= 65, f"Test case [1/3] failed: Unexpected HP {predicted_hp}"    

    # Test case 2
    print("Testing case [2/3] started.")
    predicted_hp = predict_pokemon_hp(test_data[1])
    assert 60 <= predicted_hp <= 70, f"Test case [2/3] failed: Unexpected HP {predicted_hp}"    

    # Test case 3
    print("Testing case [3/3] started.")
    predicted_hp = predict_pokemon_hp(test_data[2])
    assert 95 <= predicted_hp <= 105, f"Test case [3/3] failed: Unexpected HP {predicted_hp}"
    print("Testing finished.")

# call_test_function_line --------------------

test_predict_pokemon_hp()
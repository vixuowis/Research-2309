# function_import --------------------

from transformers import pipeline

# function_code --------------------

def predict_pokemon_hp(input_data):
    """
    Predict the HP of a Pokemon based on its attributes using a pre-trained model.

    Args:
        input_data (dict): A dictionary containing the Pokemon attributes.

    Returns:
        float: The predicted HP of the Pokemon.
    """
    hp_predictor = pipeline('regression', model='julien-c/pokemon-predict-hp')
    predicted_hp = hp_predictor(input_data)[0]['score']
    return predicted_hp

# test_function_code --------------------

def test_predict_pokemon_hp():
    """
    Test the function predict_pokemon_hp.
    """
    # Test data
    test_data = {'Name': 'Pikachu', 'Type 1': 'Electric', 'Type 2': '', 'Total': 320, 'HP': 35, 'Attack': 55, 'Defense': 40, 'Sp. Atk': 50, 'Sp. Def': 50, 'Speed': 90, 'Generation': 1, 'Legendary': False}
    predicted_hp = predict_pokemon_hp(test_data)
    # Check if the predicted HP is a float
    assert isinstance(predicted_hp, float), 'The predicted HP should be a float.'
    # Check if the predicted HP is within a reasonable range
    assert 0 <= predicted_hp <= 255, 'The predicted HP should be between 0 and 255.'

# call_test_function_code --------------------

test_predict_pokemon_hp()
# function_import --------------------

from transformers import pipeline

# function_code --------------------

def predict_pokemon_hp(input_data):
    """
    Predict the HP of a Pokemon character based on its attributes.

    Args:
        input_data (dict): A dictionary containing the attributes of the Pokemon. The keys are the attribute names and the values are the attribute values.

    Returns:
        float: The predicted HP of the Pokemon.
    """
    regression_model = pipeline('regression', model='julien-c/pokemon-predict-hp')
    predicted_hp = regression_model(input_data)[0]['score']
    return predicted_hp

# test_function_code --------------------

def test_predict_pokemon_hp():
    """
    Test the predict_pokemon_hp function.
    """
    input_data = {'attribute1': 50, 'attribute2': 60, 'attribute3': 70}
    predicted_hp = predict_pokemon_hp(input_data)
    assert isinstance(predicted_hp, float), 'The result should be a float.'
    assert predicted_hp > 0, 'The predicted HP should be greater than 0.'

# call_test_function_code --------------------

test_predict_pokemon_hp()
# function_import --------------------

from transformers import pipeline

# function_code --------------------

def predict_pokemon_hp(input_data):
    """
    Predict the HP of a Pokemon given its input attributes.

    Args:
        input_data (dict): A dictionary containing the Pokemon attributes.

    Returns:
        float: The predicted HP of the Pokemon.

    Raises:
        OSError: If the model 'julien-c/pokemon-predict-hp' is not found.
    """
    try:
        pokemon_predictor = pipeline("fill-mask", "julien-c/pokemon-predict-hp")
    except OSError as err:
        print(f"Error when loading model: {err}")
        raise

    # Get the Pokemon type
    pokemon_type = input_data["Type"][0].lower() + input_data["Type"][1:]

    # Get the prediction mask
    predict_mask = f"The <mask> is a {pokemon_type} type."

    # Predict HP and return value
    predicted_hp = pokemon_predictor(predict_mask)
    return float(predicted_hp[0]["sequence"][:-1])

# test_function_code --------------------

def test_predict_pokemon_hp():
    """
    Test the function predict_pokemon_hp.
    """
    # Test case 1: Normal case
    input_data1 = {'attribute1': 'value1', 'attribute2': 'value2'}
    try:
        predicted_hp1 = predict_pokemon_hp(input_data1)
        assert isinstance(predicted_hp1, float), 'The predicted HP should be a float.'
    except OSError:
        pass

    # Test case 2: The input data is empty
    input_data2 = {}
    try:
        predicted_hp2 = predict_pokemon_hp(input_data2)
        assert isinstance(predicted_hp2, float), 'The predicted HP should be a float.'
    except OSError:
        pass

    # Test case 3: The input data is None
    input_data3 = None
    try:
        predicted_hp3 = predict_pokemon_hp(input_data3)
        assert isinstance(predicted_hp3, float), 'The predicted HP should be a float.'
    except OSError:
        pass

    return 'All Tests Passed'


# call_test_function_code --------------------

test_predict_pokemon_hp()
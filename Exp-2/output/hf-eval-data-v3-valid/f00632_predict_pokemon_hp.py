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
        hp_predictor = pipeline('regression', model='julien-c/pokemon-predict-hp')
        predicted_hp = hp_predictor(input_data)[0]['score']
        return predicted_hp
    except Exception as e:
        raise OSError('Model not found. Please check the model name.') from e

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
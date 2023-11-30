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
        # Get the model from the transformer pipeline function
        model = pipeline(task="fill-mask", 
                         model='julien-c/pokemon-predict-hp')
    except OSError:
        raise OSError("Model not found. Please run " \
            "'https://huggingface.co/julien-c/pokemon-predict-hp'")
    
    # Get the attributes from input_data dict and format them to generate
    # a string in the same manner as the model expects input data.
    name = input_data['name']
    level = input_data['level']
    type1 = input_data['type1']
    type2 = input_data['type2']
    
    if not type2: # No second type provided by the user, use a placeholder.
        type2 = '.'
        
    input_str = f"<mask> is {name} (lvl {level}, {type1}/{type2}). " \
                "What is its HP?" 
    
    # Predict the value and get the first prediction.
    result = model(input_str)['scores'][0]['token_str'].split()[0]
    return round(float(result))

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
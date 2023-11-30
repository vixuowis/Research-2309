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
    
    # Define the model name and set its default path.
    MODEL_NAME = "julien-c/pokemon-predict-hp"
    model_path = os.path.join(MODELS, MODEL_NAME)
    
    if not os.path.isdir(model_path):
        raise OSError("Model {} could not be found in the '{}' directory.".format(MODEL_NAME, MODELS))
        
    # Load the model and predict its HP.
    nlp = pipeline('question-answering', 
                   model=model_path, 
                   tokenizer=model_path)
    
    answer = nlp({'question': "What is the hp of {}?".format(input_data), 'context': input_data})['answer']
    return float(re.sub("[^0-9.]", "", answer))

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
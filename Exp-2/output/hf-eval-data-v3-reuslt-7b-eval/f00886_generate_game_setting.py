# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_game_setting(initial_text):
    """
    Generate a game setting based on the initial text using the 'bigscience/bloom-7b1' model.

    Args:
        initial_text (str): The initial text to base the game setting on.

    Returns:
        str: The generated game setting.
    """
    # Initialize pipeline for the 'generate-from-file' task with the 'bigscience/bloom-7b1' model.
    generate_from_file = pipeline("text2text-generation",model="bigscience/bloom-7b1")
    
    return generate_from_file(initial_text,max_length=300)

# test_function_code --------------------

def test_generate_game_setting():
    """
    Test the generate_game_setting function.
    """
    assert isinstance(generate_game_setting('In a world filled with chaos and destruction'), str)
    assert isinstance(generate_game_setting('Once upon a time'), str)
    assert isinstance(generate_game_setting('In a futuristic city'), str)
    return 'All Tests Passed'


# call_test_function_code --------------------

print(test_generate_game_setting())
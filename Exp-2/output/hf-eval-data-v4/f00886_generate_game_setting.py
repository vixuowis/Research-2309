# requirements_file --------------------

!pip install -U transformers, torch

# function_import --------------------

from transformers import TextGenerationPipeline, Bloom7b1Model

# function_code --------------------

def generate_game_setting(initial_text):
    """
    Generate a setting for a new action game based on an initial text prompt.

    Parameters:
        initial_text (str): An initial text prompt for the text generation model.

    Returns:
        str: The generated text that can serve as inspiration for the game's story setting.
    """
    model = TextGenerationPipeline(model=Bloom7b1Model.from_pretrained('bigscience/bloom-7b1'))
    result = model(initial_text)
    return result[0]['generated_text']

# test_function_code --------------------

def test_generate_game_setting():
    print("Testing started.")

    # Test case 1: Starting with a theme of chaos
    print("Testing case [1/3] started.")
    theme_of_chaos = 'In a world filled with chaos and destruction, '
    setting_chaos = generate_game_setting(theme_of_chaos)
    assert type(setting_chaos) is str, "Test case [1/3] failed: The output should be a string."
    print("Test case [1/3] passed.")

    # Test case 2: Starting with a theme of adventure
    print("Testing case [2/3] started.")
    theme_of_adventure = 'An epic quest awaits the hero, '
    setting_adventure = generate_game_setting(theme_of_adventure)
    assert type(setting_adventure) is str, "Test case [2/3] failed: The output should be a string."
    print("Test case [2/3] passed.")

    # Test case 3: Starting with a theme of mystery
    print("Testing case [3/3] started.")
    theme_of_mystery = 'Hidden in the shadows, a secret world unfolds, '
    setting_mystery = generate_game_setting(theme_of_mystery)
    assert type(setting_mystery) is str, "Test case [3/3] failed: The output should be a string."
    print("Test case [3/3] passed.")
    print("Testing finished.")

test_generate_game_setting()
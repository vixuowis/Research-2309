# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_short_story(prompt):
    """
    Generates a short story based on a provided prompt using a pre-trained language model.

    Parameters:
    prompt (str): A text prompt to initiate the story generation process.

    Returns:
    str: The generated short story.
    """
    story_generator = pipeline('text-generation', model='decapoda-research/llama-7b-hf')
    story = story_generator(prompt, max_length=100)[0]['generated_text']
    return story

# test_function_code --------------------

def test_generate_short_story():
    print("Testing started.")
    prompt = "Once upon a time in a small village..."

    # Test case 1: Check if the story is generated and is of type string
    print("Testing case [1/1] started.")
    generated_story = generate_short_story(prompt)
    assert isinstance(generated_story, str), f"Test case [1/1] failed: Expected string type, got {type(generated_story)}"
    print("Test case [1/1] passed.")
    print("Testing finished.")

test_generate_short_story()
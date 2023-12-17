# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_story(starting_phrase):
    """
    Generates a story based on a given starting phrase using a text-generation pipeline.

    Args:
        starting_phrase (str): The starting phrase to seed the story generation process.

    Returns:
        str: The generated story as a continuation of the starting phrase.

    Raises:
        ValueError: If the starting_phrase is not a string or is empty.
    """
    if not isinstance(starting_phrase, str) or not starting_phrase:
        raise ValueError('The starting phrase must be a non-empty string.')

    # Initialize the text-generation pipeline with the specified model
    generator = pipeline('text-generation', model='decapoda-research/llama-13b-hf')

    # Generate the story based on the starting phrase
    generated_story = generator(starting_phrase, max_length=100, num_return_sequences=1)

    # Return the generated story text
    return generated_story[0]['generated_text']

# test_function_code --------------------

def test_generate_story():
    print("Testing started.")

    # Simulate a starting phrase
    starting_phrase = 'Once upon a time'

    # Expected output format check
    print("Testing case [1/1] started.")
    generated_story = generate_story(starting_phrase)
    assert isinstance(generated_story, str) and generated_story.startswith(starting_phrase), f"Test case [1/1] failed: The generated story should be a string and start with the starting phrase."
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_story()
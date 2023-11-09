# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_story(starting_phrase):
    """
    Generate a story based on a starting phrase using the 'decapoda-research/llama-13b-hf' model from Hugging Face Transformers.

    Args:
        starting_phrase (str): The starting phrase of the story.

    Returns:
        str: The generated story.
    """
    generator = pipeline('text-generation', model='decapoda-research/llama-13b-hf')
    generated_text = generator(starting_phrase)[0]['generated_text']
    return generated_text

# test_function_code --------------------

def test_generate_story():
    """
    Test the generate_story function.
    """
    starting_phrase = 'Once upon a time'
    generated_text = generate_story(starting_phrase)
    assert isinstance(generated_text, str), 'The output should be a string.'
    assert len(generated_text) > len(starting_phrase), 'The generated text should be longer than the starting phrase.'

# call_test_function_code --------------------

test_generate_story()
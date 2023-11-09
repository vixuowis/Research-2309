# function_import --------------------

from transformers import pipeline, set_seed

# function_code --------------------

def generate_creative_ideas(seed, prompt, max_length, num_return_sequences):
    """
    Generate creative ideas for a paragraph using the distilgpt2 model.

    Args:
        seed (int): The seed for the random number generator. This ensures reproducible results.
        prompt (str): The initial text to start the generation from.
        max_length (int): The maximum length of the generated text.
        num_return_sequences (int): The number of sequences to return.

    Returns:
        list: A list of generated text sequences.
    """
    set_seed(seed)
    generator = pipeline('text-generation', model='distilgpt2')
    creative_ideas = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
    return creative_ideas

# test_function_code --------------------

def test_generate_creative_ideas():
    """
    Test the generate_creative_ideas function.
    """
    seed = 42
    prompt = 'Once upon a time,'
    max_length = 50
    num_return_sequences = 5
    creative_ideas = generate_creative_ideas(seed, prompt, max_length, num_return_sequences)
    assert isinstance(creative_ideas, list), 'The output should be a list.'
    assert len(creative_ideas) == num_return_sequences, 'The number of returned sequences should be equal to num_return_sequences.'

# call_test_function_code --------------------

test_generate_creative_ideas()
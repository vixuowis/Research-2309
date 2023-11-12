# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_story_start(prompt: str, max_length: int = 50, num_return_sequences: int = 1):
    '''
    Generate a story start based on the given prompt using the 'sshleifer/tiny-gpt2' model.

    Args:
        prompt (str): The initial text which will be the starting point of the story.
        max_length (int, optional): The maximum length of the story. Defaults to 50.
        num_return_sequences (int, optional): The number of sequences to return. Defaults to 1.

    Returns:
        str: The generated story start.
    '''
    text_generator = pipeline('text-generation', model='sshleifer/tiny-gpt2')
    story_start = text_generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
    return story_start[0]['generated_text']

# test_function_code --------------------

def test_generate_story_start():
    '''
    Test the generate_story_start function.
    '''
    story_start = generate_story_start('A brave knight and a fearsome dragon')
    assert isinstance(story_start, str)
    assert len(story_start) > 0
    story_start = generate_story_start('A brave knight and a fearsome dragon', max_length=100)
    assert len(story_start) <= 100
    story_start = generate_story_start('A brave knight and a fearsome dragon', num_return_sequences=2)
    assert len(story_start) > 0
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_story_start()
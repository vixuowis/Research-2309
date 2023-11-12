# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_story(prompt: str, max_length: int = 500) -> str:
    """
    Generate a story based on a given prompt using the EleutherAI/gpt-j-6B model.

    Args:
        prompt (str): The initial text to start the story.
        max_length (int, optional): The maximum length of the story. Defaults to 500.

    Returns:
        str: The generated story.
    """
    text_generator = pipeline('text-generation', model='EleutherAI/gpt-j-6B')
    story_output = text_generator(prompt, max_length=max_length)
    story = story_output[0]['generated_text']
    return story

# test_function_code --------------------

def test_generate_story():
    """
    Test the generate_story function.
    """
    story_prompt = 'Write a story about a spaceship journey to a distant planet in search of a new home for humanity.'
    story = generate_story(story_prompt)
    assert isinstance(story, str), 'The output should be a string.'
    assert len(story) <= 500, 'The length of the story should not exceed the maximum length.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_story()
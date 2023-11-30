# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_story(prompt: str) -> str:
    """
    Generate a short story based on a given prompt using the LLaMA-7B language model.

    Args:
        prompt (str): The initial prompt to base the story on.

    Returns:
        str: The generated story.

    Raises:
        OSError: If the specified model is not found.
    """

    try:
        # Load pipeline
        nlp = pipeline(task="text-generation", model='./lama')

        # Generate story
        generated = nlp(prompt)[0]['generated_text']
        
        return generated.strip()

    except OSError as e:
        raise OSError("Model not found. Please ensure that it has been downloaded from S3.") from e

# test_function_code --------------------

def test_generate_story():
    """
    Test the generate_story function.
    """
    try:
        # Test with a simple prompt
        prompt = 'Once upon a time in a small village...'
        story = generate_story(prompt)
        assert isinstance(story, str)

        # Test with a different prompt
        prompt = 'In a galaxy far, far away...'
        story = generate_story(prompt)
        assert isinstance(story, str)

        # Test with an empty prompt
        prompt = ''
        story = generate_story(prompt)
        assert isinstance(story, str)

        print('All Tests Passed')
    except OSError:
        print('Model not found. Skipping tests.')


# call_test_function_code --------------------

test_generate_story()
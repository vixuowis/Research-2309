# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_space_journey_story(prompt, max_length=500):
    """
    Generates a story about a spaceship journey to a distant planet in search of a new home for humanity.

    Args:
        prompt (str): The prompt to feed to the text generation model.
        max_length (int): The maximum length of the generated story text.

    Returns:
        str: The generated story text.

    Raises:
        RuntimeError: If there is an issue with model loading or text generation.
    """
    try:
        # Load the text generation model
        text_generator = pipeline('text-generation', model='EleutherAI/gpt-j-6B')
        # Generate the story with the given prompt
        story_output = text_generator(prompt, max_length=max_length)
        story = story_output[0]['generated_text']
        return story
    except Exception as e:
        raise RuntimeError(f"An error occurred during text generation: {e}")

# test_function_code --------------------

def test_generate_space_journey_story():
    print("Testing started.")
    
    # Test case 1: Ensure that the function returns a string
    print("Testing case [1/2] started.")
    prompt = "Write a story about a spaceship journey to a distant planet in search of a new home for humanity."
    result = generate_space_journey_story(prompt)
    assert isinstance(result, str), f"Test case [1/2] failed: Expected a string, got {type(result)}"
    
    # Test case 2: Ensure that the function raises an error when an invalid model name is provided
    print("Testing case [2/2] started.")
    invalid_prompt = "This is an invalid prompt that should not work."
    try:
        generate_space_journey_story(invalid_prompt, model='invalid-model-name')
        assert False, "Test case [2/2] failed: Expected RuntimeError"
    except RuntimeError:
        pass
    
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_space_journey_story()
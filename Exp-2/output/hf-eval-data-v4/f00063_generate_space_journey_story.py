# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_space_journey_story(prompt):
    """
    Generates a story about a spaceship journey to a distant planet in search of a new home for humanity.

    Parameters:
        prompt (str): A story prompt to feed the text generation model.

    Returns:
        str: A generated story based on the input prompt.
    """
    text_generator = pipeline('text-generation', model='EleutherAI/gpt-j-6B')
    story_output = text_generator(prompt, max_length=500)
    story = story_output[0]['generated_text']
    return story

# test_function_code --------------------

def test_generate_space_journey_story():
    print("Testing started.")
    sample_prompt = "The spaceship ventured into the unknown, its crew eager to find a new planet for humanity."

    # Just testing if the function returns a string of reasonable length
    print("Testing case [1/1] started.")
    story = generate_space_journey_story(sample_prompt)
    assert isinstance(story, str) and len(story) > 100, "Test case [1/1] failed: Story output is not a string or is too short."
    print("Testing case [1/1] passed.")
    print("Testing finished.")

# Run the test function
test_generate_space_journey_story()
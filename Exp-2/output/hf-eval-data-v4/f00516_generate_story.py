# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_story(starting_phrase):
    """
    Generate a story based on the provided starting phrase.

    Parameters:
        starting_phrase (str): The initial phrase to start the story.

    Returns:
        str: The generated text forming a story.
    """
    generator = pipeline('text-generation', model='decapoda-research/llama-13b-hf')
    generated_text = generator(starting_phrase, max_length=200)[0]['generated_text']
    return generated_text


# test_function_code --------------------

def test_generate_story():
    print("Testing generate_story function.")
    starting_phrases = [
        'Once upon a time', 
        'In a galaxy far, far away', 
        'At the dawn of time'
    ]

    for index, phrase in enumerate(starting_phrases):
        print(f"Testing case [{index+1}/{len(starting_phrases)}] started.")
        story = generate_story(phrase)
        assert isinstance(story, str), f"Test case [{index+1}/{len(starting_phrases)}] failed: The generated story is not a string."
        assert story.startswith(phrase), f"Test case [{index+1}/{len(starting_phrases)}] failed: The story does not start with the given phrase."
        print(f"Test case [{index+1}/{len(starting_phrases)}] passed.")
    print("All test cases passed.")

# Running the test function
test_generate_story()
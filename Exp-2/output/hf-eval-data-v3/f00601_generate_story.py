# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_story(short_description):
    """
    Generate a creative story based on a short description using a pre-trained model.

    Args:
        short_description (str): A short description to base the story on.

    Returns:
        str: A generated story based on the provided short description.
    """
    story_generator = pipeline('text-generation', model='decapoda-research/llama-7b-hf')
    generated_story = story_generator(short_description)[0]['generated_text']
    return generated_story

# test_function_code --------------------

def test_generate_story():
    """
    Test the generate_story function.
    """
    test_description_1 = 'In a world where digital art comes to life...'
    test_description_2 = 'Once upon a time in a faraway land...'
    test_description_3 = 'In the midst of a global pandemic...'

    assert isinstance(generate_story(test_description_1), str)
    assert isinstance(generate_story(test_description_2), str)
    assert isinstance(generate_story(test_description_3), str)

    print('All Tests Passed')

# call_test_function_code --------------------

test_generate_story()
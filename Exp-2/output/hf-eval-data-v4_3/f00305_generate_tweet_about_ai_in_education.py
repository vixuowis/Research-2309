# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_tweet_about_ai_in_education(topic):
    """
    Generates a tweet on the topic of AI in education using a pre-trained language model.

    Args:
        topic (str): The topic to generate a tweet about.

    Returns:
        str: A generated tweet text.

    Raises:
        ValueError: If the input topic is an empty string.
    """
    if topic == '':
        raise ValueError('Input topic cannot be an empty string.')
    generator = pipeline('text-generation', model='bigscience/bloom-560m')
    result = generator(topic, max_length=280)
    return result[0]['generated_text']

# test_function_code --------------------

def test_generate_tweet_about_ai_in_education():
    print("Testing started.")

    # Test case 1: Standard topic
    print("Testing case [1/3] started.")
    tweet = generate_tweet_about_ai_in_education('The Future of AI in Education')
    assert isinstance(tweet, str), f"Test case [1/3] failed: The result should be a string."

    # Test case 2: Empty topic
    print("Testing case [2/3] started.")
    try:
        generate_tweet_about_ai_in_education('')
        assert False, "Test case [2/3] failed: Should raise ValueError for empty topic."
    except ValueError:
        assert True

    # Test case 3: Long topic (truncated by model)
    print("Testing case [3/3] started.")
    long_topic = 'A' * 300
    tweet = generate_tweet_about_ai_in_education(long_topic)
    assert len(tweet) <= 280 and isinstance(tweet, str), f"Test case [3/3] failed: The result should be a string and truncated to tweet length."
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_tweet_about_ai_in_education()
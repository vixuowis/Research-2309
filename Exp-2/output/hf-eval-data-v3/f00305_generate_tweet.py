# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_tweet(topic):
    '''
    Generate a tweet on a given topic using the 'bigscience/bloom-560m' model.

    Args:
        topic (str): The topic of the tweet.

    Returns:
        str: The generated tweet.
    '''
    generator = pipeline('text-generation', model='bigscience/bloom-560m')
    tweet = generator(topic, max_length=280)
    return tweet[0]['generated_text']

# test_function_code --------------------

def test_generate_tweet():
    '''
    Test the generate_tweet function.
    '''
    tweet1 = generate_tweet('The Future of AI in Education')
    assert isinstance(tweet1, str)
    assert len(tweet1) <= 280

    tweet2 = generate_tweet('Climate Change and Technology')
    assert isinstance(tweet2, str)
    assert len(tweet2) <= 280

    tweet3 = generate_tweet('The Impact of Blockchain on Finance')
    assert isinstance(tweet3, str)
    assert len(tweet3) <= 280

    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_tweet()
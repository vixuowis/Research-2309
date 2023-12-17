# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_tweet_about_ai_in_education():
    # This function uses a pre-trained model to generate a tweet about the future of AI in Education.
    generator = pipeline('text-generation', model='bigscience/bloom-560m')
    topic = 'The Future of AI in Education'
    tweet = generator(topic, max_length=280)[0]['generated_text']
    return tweet

# test_function_code --------------------

def test_generate_tweet_about_ai_in_education():
    print('Testing generate_tweet_about_ai_in_education function...')
    output = generate_tweet_about_ai_in_education()
    assert isinstance(output, str), f'The function should return a string, but got {type(output)}'
    assert len(output) <= 280, 'The length of the tweet should not exceed 280 characters.'
    print('All tests passed!')
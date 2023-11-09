from transformers import pipeline


def generate_tweet(topic):
    """
    This function generates a tweet on a given topic using the 'bigscience/bloom-560m' model from the Transformers library.
    
    Parameters:
    topic (str): The topic of the tweet.
    
    Returns:
    str: The generated tweet.
    """
    # Load the 'bigscience/bloom-560m' model
    generator = pipeline('text-generation', model='bigscience/bloom-560m')
    
    # Generate a tweet on the given topic
    tweet = generator(topic, max_length=280)
    
    return tweet[0]['generated_text']
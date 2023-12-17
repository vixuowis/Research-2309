# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_customer_message_sentiment(message):
    """
    Analyze the sentiment of a customer support message.

    Args:
        message (str): A message from customer support chat system.

    Returns:
        dict: A dictionary containing the sentiment analysis result.

    Raises:
        ValueError: If the message is empty or None.
    """
    if not message:
        raise ValueError('The message is empty or None')
    sentiment_task = pipeline('sentiment-analysis', model='cardiffnlp/twitter-xlm-roberta-base-sentiment')
    result = sentiment_task(message)
    return result

# test_function_code --------------------

def test_analyze_customer_message_sentiment():
    print("Testing started.")
    # Test case 1: Positive sentiment
    print("Testing case [1/3] started.")
    positive_message = "I'm really happy with the service!"
    positive_result = analyze_customer_message_sentiment(positive_message)
    assert positive_result[0]['label'] == 'POSITIVE', f"Test case [1/3] failed: {positive_result}"
    
    # Test case 2: Negative sentiment
    print("Testing case [2/3] started.")
    negative_message = "I'm really frustrated with the service."
    negative_result = analyze_customer_message_sentiment(negative_message)
    assert negative_result[0]['label'] == 'NEGATIVE', f"Test case [2/3] failed: {negative_result}"

    # Test case 3: Neutral sentiment
    print("Testing case [3/3] started.")
    neutral_message = "The service is okay."
    neutral_result = analyze_customer_message_sentiment(neutral_message)
    assert neutral_result[0]['label'] == 'NEUTRAL', f"Test case [3/3] failed: {neutral_result}"
    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_customer_message_sentiment()
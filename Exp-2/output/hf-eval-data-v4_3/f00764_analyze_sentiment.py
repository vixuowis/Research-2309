# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(text):
    """
    Performs sentiment analysis on the given text using a pre-trained model.

    Args:
        text (str): The text to analyze sentiment for.

    Returns:
        dict: The result of sentiment analysis containing label and score.

    Raises:
        ValueError: If the text is empty.
    """
    if not text:
        raise ValueError('The text for sentiment analysis is empty.')
    # Initialize the sentiment analysis pipeline
    sentiment_task = pipeline('sentiment-analysis', model='cardiffnlp/twitter-xlm-roberta-base-sentiment')
    # Perform sentiment analysis
    return sentiment_task(text)

# test_function_code --------------------

def test_analyze_sentiment():
    print("Testing started.")
    # Test cases
    positive_text = 'I really enjoyed the experience at this store.'
    negative_text = 'This is the worst product I have ever bought.'
    neutral_text = 'The product is ok, nothing special.'

    # Testing case 1: Positive sentiment
    print("Testing case [1/3] started.")
    result = analyze_sentiment(positive_text)
    assert result[0]['label'] == 'LABEL_2', f"Test case [1/3] failed: {result}"  # Assuming LABEL_2 is positive

    # Testing case 2: Negative sentiment
    print("Testing case [2/3] started.")
    result = analyze_sentiment(negative_text)
    assert result[0]['label'] == 'LABEL_0', f"Test case [2/3] failed: {result}"  # Assuming LABEL_0 is negative

    # Testing case 3: Neutral sentiment
    print("Testing case [3/3] started.")
    result = analyze_sentiment(neutral_text)
    assert result[0]['label'] == 'LABEL_1', f"Test case [3/3] failed: {result}"  # Assuming LABEL_1 is neutral
    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_sentiment()
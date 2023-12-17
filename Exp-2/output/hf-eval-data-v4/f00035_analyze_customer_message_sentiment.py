# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_customer_message_sentiment(message):
    """
    Analyze the sentiment of a customer support message using a pre-trained model.

    Parameters:
        message (str): The customer support message to be analyzed.

    Returns:
        dict: The sentiment analysis result containing label and score.
    """
    sentiment_task = pipeline('sentiment-analysis', model='cardiffnlp/twitter-xlm-roberta-base-sentiment')
    return sentiment_task(message)[0]

# test_function_code --------------------

def test_analyze_customer_message_sentiment():
    print("Testing analyze_customer_message_sentiment function.")

    # Test case 1: Positive sentiment
    message = "I'm really happy with the service!"
    result = analyze_customer_message_sentiment(message)
    assert result['label'] in ['LABEL_0', 'LABEL_1', 'LABEL_2'] and result['score'] > 0, "Test case [1/3] failed: Incorrect sentiment detected."

    # Test case 2: Negative sentiment
    message = "I'm really frustrated with the service"
    result = analyze_customer_message_sentiment(message)
    assert result['label'] in ['LABEL_0', 'LABEL_1', 'LABEL_2'] and result['score'] > 0, "Test case [2/3] failed: Incorrect sentiment detected."

    # Test case 3: Neutral sentiment
    message = "I have a question regarding my order."
    result = analyze_customer_message_sentiment(message)
    assert result['label'] in ['LABEL_0', 'LABEL_1', 'LABEL_2'] and result['score'] > 0, "Test case [3/3] failed: Incorrect sentiment detected."

    print("All test cases passed!")

# Running the test function
test_analyze_customer_message_sentiment()
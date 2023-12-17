# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_financial_sentiment(text):
    """
    Analyze the sentiment of a financial text using the FinBERT model.

    :param text: A string containing the financial text to be analyzed
    :return: A dictionary containing the sentiment classification result
    """
    # Load the FinBERT sentiment analysis model
    classifier = pipeline('sentiment-analysis', model='ProsusAI/finbert')

    # Analyze the sentiment of the input text
    result = classifier(text)

    # Return the result
    return result

# test_function_code --------------------

def test_analyze_financial_sentiment():
    print("Testing started.")
    test_text = 'I love this financial service app. It has made managing my finances so much easier!'
    expected_sentiment = 'positive'

    # Test the sentiment analysis function
    print("Testing sentiment analysis started.")
    result = analyze_financial_sentiment(test_text)
    assert result[0]['label'].lower() == expected_sentiment, f"Test failed: Expected sentiment {expected_sentiment}, got {result[0]['label']}"

    print("Testing finished.")

# Run the test function
test_analyze_financial_sentiment()
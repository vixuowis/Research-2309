# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(text):
    sentiment_analysis = pipeline('sentiment-analysis', model='siebert/sentiment-roberta-large-english')
    return sentiment_analysis(text)


# test_function_code --------------------

def test_analyze_sentiment():
    print("Testing analyze_sentiment.")
    # Test case 1: Positive sentiment
    positive_text = "I absolutely love this product!"
    result_positive = analyze_sentiment(positive_text)
    assert 'label' in result_positive[0] and result_positive[0]['label'] == 'POSITIVE', "Positive sentiment detected incorrectly."
    print("Test case 1 passed.")

    # Test case 2: Negative sentiment
    negative_text = "This is the worst product ever."
    result_negative = analyze_sentiment(negative_text)
    assert 'label' in result_negative[0] and result_negative[0]['label'] == 'NEGATIVE', "Negative sentiment detected incorrectly."
    print("Test case 2 passed.")

    # Test case 3: Neutral case (not applicable for binary sentiment analysis)
    # Since the model only predicts positive or negative, we will not have a neutral case in this scenario.
    print("Testing completed.")
test_analyze_sentiment()

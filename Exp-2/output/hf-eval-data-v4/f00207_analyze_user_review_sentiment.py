# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_user_review_sentiment(review_text):
    sentiment_analyzer = pipeline('sentiment-analysis', model='finiteautomata/beto-sentiment-analysis')
    sentiment_result = sentiment_analyzer(review_text)
    return sentiment_result

# test_function_code --------------------

def test_analyze_user_review_sentiment():
    print("Testing analyze_user_review_sentiment function.")

    # Test case 1: Positive review
    review_text = 'El mejor app que he usado hasta ahora. Genial!'
    expected_sentiment = 'POS'
    result = analyze_user_review_sentiment(review_text)
    assert result[0]['label'] == expected_sentiment, f"Test case failed: expected {expected_sentiment}, got {result[0]['label']}"

    # Test case 2: Negative review
    review_text = 'La peor experiencia de mi vida, no la recomiendo en absoluto.'
    expected_sentiment = 'NEG'
    result = analyze_user_review_sentiment(review_text)
    assert result[0]['label'] == expected_sentiment, f"Test case failed: expected {expected_sentiment}, got {result[0]['label']}"

    # Test case 3: Neutral review
    review_text = 'Es una app mais, nada especial.'
    expected_sentiment = 'NEU'
    result = analyze_user_review_sentiment(review_text)
    assert result[0]['label'] == expected_sentiment, f"Test case failed: expected {expected_sentiment}, got {result[0]['label']}"

    print("All test cases passed!")

test_analyze_user_review_sentiment()
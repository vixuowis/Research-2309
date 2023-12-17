# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def detect_low_rated_reviews(text):
    sentiment_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    result = sentiment_pipeline(text)
    if int(result[0]['label'][-1]) < 3:
        return True
    return False

# test_function_code --------------------

def test_detect_low_rated_reviews():
    print("Testing started.")
    test_reviews = [
        ('I love this product!', False),
        ('This is the worst thing I have ever bought.', True),
        ('Not worth the price.', True)
    ]
    for i, (review, expected) in enumerate(test_reviews, 1):
        result = detect_low_rated_reviews(review)
        assert result == expected, f"Test case [{i}/3] failed: Expected {expected}, got {result}"
        print(f"Testing case [{i}/3] succeeded.")
    print("Testing finished.")

test_detect_low_rated_reviews()
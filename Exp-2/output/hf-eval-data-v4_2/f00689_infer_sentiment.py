# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def infer_sentiment(comment: str) -> dict:
    """
    Infers the sentiment of a given user comment using zero-shot classification.

    Args:
        comment (str): The user comment text to analyze.

    Returns:
        dict: A dictionary containing the top sentiment label and its confidence score.

    Raises:
        ValueError: If the input comment is empty.

    """
    if not comment:
        raise ValueError('Input comment is empty')
    nlp = pipeline('zero-shot-classification', model='valhalla/distilbart-mnli-12-6')
    result = nlp(comment, ['positive', 'negative'])
    sentiment = result['labels'][0]
    confidence = result['scores'][0]
    return {'sentiment': sentiment, 'confidence': confidence}

# test_function_code --------------------

def test_infer_sentiment():
    print("Testing started.")
    test_cases = [
        ('I love this product, it is amazing!', 'positive'),
        ('This is the worst purchase I have ever made.', 'negative'),
        ('', ValueError)
    ]

    for i, (comment, expected) in enumerate(test_cases):
        print(f"Testing case [{i+1}/{len(test_cases)}] started.")
        if expected is ValueError:
            try:
                _ = infer_sentiment(comment)
                assert False, f"Test case [{i+1}/{len(test_cases)}] failed: Expected ValueError"
            except ValueError:
                pass  # Expected exception
        else:
            result = infer_sentiment(comment)
            assert result['sentiment'] == expected, f"Test case [{i+1}/{len(test_cases)}] failed: Expected {expected} sentiment"
    print("Testing finished.")

# call_test_function_line --------------------

test_infer_sentiment()
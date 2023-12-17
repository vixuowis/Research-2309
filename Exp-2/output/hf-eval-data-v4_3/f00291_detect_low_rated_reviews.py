# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def detect_low_rated_reviews(text):
    """
    Detects low-rated product reviews based on sentiment analysis.

    Args:
        text (str): A product review text in English, Dutch, German, French, Italian, or Spanish.

    Returns:
        bool: True if the review is low-rated, False otherwise.

    Raises:
        ValueError: If the input text is not a string or is empty.
    """
    if not isinstance(text, str) or not text:
        raise ValueError('The input text should be a non-empty string.')

    sentiment_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    result = sentiment_pipeline(text)
    return int(result[0]['label'][-1]) < 3

# test_function_code --------------------

def test_detect_low_rated_reviews():
    print("Testing started.")
    # Test cases with various languages and expected outcomes
    test_cases = [
        ("This product is terrible!", True),
        ("Really enjoyed this, excellent!", False),
        ("Nicht gut, nicht kaufen.", True),
        ("C'est parfait, j'adore!", False),
        ("Producto defectuoso, no recomiendo.", True),
        ("È un prodotto di qualità.", False)
    ]

    for i, (review, expected) in enumerate(test_cases, 1):
        print(f"Testing case [{i}/{len(test_cases)}] started.")
        assert detect_low_rated_reviews(review) == expected, f"Test case [{i}/{len(test_cases)}] failed: the sentiment analysis did not match the expected output."
    print("Testing finished.")

# run the test function
test_detect_low_rated_reviews()

# call_test_function_line --------------------

test_detect_low_rated_reviews()
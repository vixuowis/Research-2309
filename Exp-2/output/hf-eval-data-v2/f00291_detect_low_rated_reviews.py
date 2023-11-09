# function_import --------------------

from transformers import pipeline

# function_code --------------------

def detect_low_rated_reviews(review_text):
    """
    Detects low-rated product reviews in six languages: English, Dutch, German, French, Italian, and Spanish.

    Args:
        review_text (str): The text of the product review.

    Returns:
        bool: True if the review is low-rated (less than 3 stars), False otherwise.
    """
    sentiment_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    result = sentiment_pipeline(review_text)
    return int(result[0]['label'][-1]) < 3

# test_function_code --------------------

def test_detect_low_rated_reviews():
    """
    Tests the function detect_low_rated_reviews.
    """
    assert detect_low_rated_reviews('I love this product!') == False
    assert detect_low_rated_reviews('This product is terrible!') == True
    assert detect_low_rated_reviews('Dit product is verschrikkelijk!') == True # Dutch
    assert detect_low_rated_reviews('Dieses Produkt ist schrecklich!') == True # German
    assert detect_low_rated_reviews('Ce produit est terrible!') == True # French
    assert detect_low_rated_reviews('Questo prodotto è terribile!') == True # Italian
    assert detect_low_rated_reviews('¡Este producto es terrible!') == True # Spanish

# call_test_function_code --------------------

test_detect_low_rated_reviews()
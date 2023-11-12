# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def analyze_customer_reviews(reviews, seed_phrases):
    """
    Analyze customer reviews using SentenceTransformer model.

    Args:
        reviews (list): A list of customer reviews.
        seed_phrases (dict): A dictionary with keys as sentiments (positive, neutral, negative) and values as lists of seed phrases.

    Returns:
        dict: A dictionary with keys as sentiments and values as lists of reviews belonging to that sentiment.

    Raises:
        ValueError: If reviews or seed_phrases is not a list or dict respectively.
    """
    if not isinstance(reviews, list) or not isinstance(seed_phrases, dict):
        raise ValueError('Reviews should be a list and seed_phrases should be a dictionary.')

    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
    review_embeddings = model.encode(reviews)

    sentiment_analysis_result = {}
    for sentiment, phrases in seed_phrases.items():
        phrase_embeddings = model.encode(phrases)
        for review_embedding in review_embeddings:
            similarities = cosine_similarity([review_embedding], phrase_embeddings)
            if similarities.max() > 0.8:
                if sentiment not in sentiment_analysis_result:
                    sentiment_analysis_result[sentiment] = []
                sentiment_analysis_result[sentiment].append(reviews[review_embeddings.index(review_embedding)])

    return sentiment_analysis_result

# test_function_code --------------------

def test_analyze_customer_reviews():
    """Test the analyze_customer_reviews function."""
    reviews = ['I love this product!', 'This is okay.', 'I hate this!']
    seed_phrases = {
        'positive': ['This is great', 'I love it'],
        'neutral': ['This is okay', 'It is fine'],
        'negative': ['I hate this', 'This is terrible']
    }
    result = analyze_customer_reviews(reviews, seed_phrases)
    assert 'positive' in result and 'I love this product!' in result['positive'], 'Positive review classification failed.'
    assert 'neutral' in result and 'This is okay.' in result['neutral'], 'Neutral review classification failed.'
    assert 'negative' in result and 'I hate this!' in result['negative'], 'Negative review classification failed.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_analyze_customer_reviews()
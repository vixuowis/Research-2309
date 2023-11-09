# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def analyze_customer_reviews(customer_reviews, seed_phrases):
    """
    Analyze customer reviews to determine their sentiment (positive, neutral, negative).

    Args:
        customer_reviews (list of str): The customer reviews to analyze.
        seed_phrases (dict): A dictionary where the keys are sentiments (positive, neutral, negative) and the values are lists of seed phrases representing each sentiment.

    Returns:
        dict: A dictionary where the keys are the customer reviews and the values are the determined sentiments.
    """
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
    review_embeddings = model.encode(customer_reviews)
    sentiment_analysis_result = {}
    for review, embedding in zip(customer_reviews, review_embeddings):
        similarities = {}
        for sentiment, phrases in seed_phrases.items():
            phrase_embeddings = model.encode(phrases)
            similarity = cosine_similarity([embedding], phrase_embeddings).mean()
            similarities[sentiment] = similarity
        sentiment_analysis_result[review] = max(similarities, key=similarities.get)
    return sentiment_analysis_result

# test_function_code --------------------

def test_analyze_customer_reviews():
    """
    Test the analyze_customer_reviews function.
    """
    customer_reviews = ['I love this product!', 'This product is okay.', 'I hate this product!']
    seed_phrases = {
        'positive': ['I love this', 'This is great'],
        'neutral': ['This is okay', 'I feel indifferent about this'],
        'negative': ['I hate this', 'This is terrible']
    }
    result = analyze_customer_reviews(customer_reviews, seed_phrases)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(customer_reviews)
    assert set(result.values()).issubset(set(seed_phrases.keys()))

# call_test_function_code --------------------

test_analyze_customer_reviews()
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def analyze_customer_reviews(customer_reviews: list, seed_phrases: dict) -> dict:
    """
    Analyze customer reviews using SentenceTransformer model.

    Args:
        customer_reviews (list): A list of customer reviews to be analyzed.
        seed_phrases (dict): A dictionary of seed phrases with known sentiment. Keys are sentiments (positive, neutral, negative), values are lists of phrases.

    Returns:
        dict: A dictionary where keys are sentiments (positive, neutral, negative), and values are lists of reviews that belong to each sentiment.
    """
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
    review_embeddings = model.encode(customer_reviews)
    sentiment_analysis_result = {}

    for sentiment, phrases in seed_phrases.items():
        phrase_embeddings = model.encode(phrases)
        for review_embedding in review_embeddings:
            similarities = cosine_similarity([review_embedding], phrase_embeddings)
            if np.max(similarities) > 0.8:
                if sentiment not in sentiment_analysis_result:
                    sentiment_analysis_result[sentiment] = []
                sentiment_analysis_result[sentiment].append(customer_reviews[review_embeddings.index(review_embedding)])

    return sentiment_analysis_result
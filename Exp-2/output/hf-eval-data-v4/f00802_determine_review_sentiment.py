# requirements_file --------------------

!pip install -U sentence-transformers numpy scikit-learn

# function_import --------------------

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def determine_review_sentiment(reviews):
    # Instantiate the sentence transformer model
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')

    # Define seed phrases with known sentiment
    seed_phrases = {
        'positive': 'I love this product!',
        'neutral': 'This product is okay.',
        'negative': 'I dislike this product.'
    }

    # Encode the seed phrases
    seed_embeddings = model.encode(list(seed_phrases.values()))

    # Convert reviews to embeddings using the model
    review_embeddings = model.encode(reviews)

    # Determine sentiment by comparing embeddings
    sentiment_result = {}
    for review, embedding in zip(reviews, review_embeddings):
        similarities = cosine_similarity([embedding], seed_embeddings)[0]
        sentiment = max(seed_phrases, key=lambda k: similarities[list(seed_phrases.values()).index(seed_phrases[k])])
        sentiment_result[review] = sentiment

    return sentiment_result


# test_function_code --------------------

def test_determine_review_sentiment():
    print("Testing determine_review_sentiment function.")
    sample_reviews = [
        'This product made my day!',
        'It does the job.',
        'Not what I expected at all.',
        'Could be better.',
        'Highly recommend it!'
    ]

    # Expected sentiments are positive, neutral, negative, negative, and positive
    expected_sentiments = ['positive', 'neutral', 'negative', 'negative', 'positive']

    # Run the function
    sentiments = determine_review_sentiment(sample_reviews)

    # Test
    for review, expected_sentiment in zip(sample_reviews, expected_sentiments):
        assert sentiments[review] == expected_sentiment, f"Sentiment for review '{review}' was incorrectly determined as {sentiments[review]}. Expected {expected_sentiment}."

    print("All test cases passed!")

# Run the test function
test_determine_review_sentiment()

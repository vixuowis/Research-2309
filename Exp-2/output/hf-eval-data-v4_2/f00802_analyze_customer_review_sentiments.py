# requirements_file --------------------

pip install -U sentence-transformers

# function_import --------------------

from sentence_transformers import SentenceTransformer, util

# function_code --------------------

def analyze_customer_review_sentiments(reviews, seed_phrases):
    """
    Analyze sentiments of customer reviews.

    Args:
        reviews (List[str]): A list of customer reviews to be analyzed.
        seed_phrases (Dict[str, List[str]]): A dictionary containing lists of seed
            phrases for each sentiment category ('positive', 'neutral', 'negative').

    Returns:
        Dict[str, List[str]]: A dictionary with keys as sentiments and a list of
          reviews that fall under each sentiment category.

    Raises:
        ValueError: If 'reviews' is not a list or 'seed_phrases' is not a dictionary.
    """
    if not isinstance(reviews, list) or not isinstance(seed_phrases, dict):
        raise ValueError("'reviews' must be a list and 'seed_phrases' must be a dictionary.")

    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
    review_embeddings = model.encode(reviews)
    sentiment_categories = {sentiment: [] for sentiment in seed_phrases}

    for review, embedding in zip(reviews, review_embeddings):
        max_similarity = 0
        max_category = None
        for category, phrases in seed_phrases.items():
            category_embeddings = model.encode(phrases)
            similarities = util.cos_sim(embedding, category_embeddings)
            max_sim_for_category = max(similarities)
            if max_sim_for_category > max_similarity:
                max_similarity = max_sim_for_category
                max_category = category

        sentiment_categories[max_category].append(review)

    return sentiment_categories

# test_function_code --------------------

def test_analyze_customer_review_sentiments():
    print("Testing started.")
    reviews = [
        "I absolutely love this product! Best purchase ever!",
        "It's okay, but I expected more.",
        "Terrible experience, will not buy again."]
    seed_phrases = {
        'positive': ['This is great', 'Love this', 'Happy with the purchase'],
        'neutral': ['It is okay', 'Average product', 'Nothing special'],
        'negative': ['Disappointed', 'Could be better', 'Not what I expected']
    }

    expected_output = {
        'positive': [reviews[0]],
        'neutral': [reviews[1]],
        'negative': [reviews[2]]
    }

    # Test case
    print("Testing case [1/1] started.")
    result = analyze_customer_review_sentiments(reviews, seed_phrases)
    assert result == expected_output, f"Test case [1/1] failed: {result}"
    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_customer_review_sentiments()
# requirements_file --------------------

!pip install -U sentence-transformers sklearn

# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_similarity(breaking_news_text, other_texts):
    """
    Calculate the similarity between a breaking news article and a list of other articles.

    Parameters:
    breaking_news_text (str): The main text of the breaking news article.
    other_texts (List[str]): A list of texts from other articles.

    Returns:
    List[float]: A list of similarity scores between the breaking news article and each of the other articles.
    """
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    texts = [breaking_news_text] + other_texts
    embeddings = model.encode(texts)
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix[0, 1:].tolist()

# test_function_code --------------------

def test_calculate_similarity():
    breaking_news = "This is a breaking news story about an important event."
    other_articles = [
        "An important event has just occurred and here are the details.",
        "This story covers the recent event that took place.",
        "Completely unrelated news article for testing dissimilarity."
    ]

    similarities = calculate_similarity(breaking_news, other_articles)
    assert len(similarities) == 3, "The function should return a list of 3 similarity scores."
    assert all(isinstance(score, float) for score in similarities), "All similarity scores should be floats."
    print("All test cases passed for calculate_similarity function.")

# Test function call
test_calculate_similarity()
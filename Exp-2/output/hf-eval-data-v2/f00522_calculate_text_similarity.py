# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_text_similarity(breaking_news_text, other_article_texts):
    """
    Calculate the similarity between the main text of a breaking news article and other articles.

    Args:
        breaking_news_text (str): The main text of the breaking news article.
        other_article_texts (list): A list of texts from other articles.

    Returns:
        breaking_news_similarities (list): A list of similarity scores between the breaking news article and other articles.
    """
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    texts = [breaking_news_text] + other_article_texts
    embeddings = model.encode(texts)
    similarity_matrix = cosine_similarity(embeddings)
    breaking_news_similarities = similarity_matrix[0, 1:]
    return breaking_news_similarities

# test_function_code --------------------

def test_calculate_text_similarity():
    """
    Test the function calculate_text_similarity.
    """
    breaking_news_text = 'This is a breaking news article.'
    other_article_texts = ['This is another article.', 'This is yet another article.']
    similarities = calculate_text_similarity(breaking_news_text, other_article_texts)
    assert len(similarities) == len(other_article_texts), 'The number of similarity scores should be equal to the number of other articles.'
    assert all(0 <= score <= 1 for score in similarities), 'All similarity scores should be between 0 and 1.'

# call_test_function_code --------------------

test_calculate_text_similarity()
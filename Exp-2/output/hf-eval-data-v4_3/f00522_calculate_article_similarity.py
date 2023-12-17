# requirements_file --------------------

import subprocess

requirements = ["sentence-transformers", "sklearn"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_article_similarity(breaking_news_text, other_articles):
    """
    Calculates the similarity between the main text of a breaking news article
    and a list of other articles using a pre-trained multilingual model.

    Args:
        breaking_news_text (str): The main text of the breaking news article.
        other_articles (list of str): A list containing the texts of other articles.

    Returns:
        list: A list containing similarity scores between the breaking news
              article and each of the other articles.

    Raises:
        ValueError: If any of the input texts is not a string or if the
                    other_articles list is empty.
    """
    if not isinstance(breaking_news_text, str):
        raise ValueError("Breaking news text must be a string.")
    if not all(isinstance(article, str) for article in other_articles):
        raise ValueError("All other articles must be strings.")
    if len(other_articles) == 0:
        raise ValueError("The list of other articles cannot be empty.")

    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    texts = [breaking_news_text] + other_articles
    embeddings = model.encode(texts)
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix[0, 1:].tolist()

# test_function_code --------------------

def test_calculate_article_similarity():
    print("Testing started.")
    breaking_news_text = "Breaking news: Scientists discover water on Mars."
    other_articles = [
        "Astronomers found liquid water on Mars.",
        "Political update: New policies introduced.",
        "Celebrity interview reveals unexpected hobbies."
    ]

    print("Testing case [1/3] started.")
    similarities = calculate_article_similarity(breaking_news_text, other_articles)
    assert all(0 <= score <= 1 for score in similarities), f"Test case [1/3] failed: Similarity scores out of range: {similarities}"

    print("Testing case [2/3] started.")
    assert len(similarities) == len(other_articles), f"Test case [2/3] failed: Number of similarity scores doesn't match number of articles."

    print("Testing case [3/3] started.")
    assert isinstance(similarities, list), f"Test case [3/3] failed: Expected a list of similarities, got {type(similarities).__name__}"
    print("Testing finished.")

# call_test_function_line --------------------

test_calculate_article_similarity()
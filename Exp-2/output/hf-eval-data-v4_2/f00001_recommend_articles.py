# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# function_code --------------------

def recommend_articles(user_likes, new_article, threshold=0.8):
    """
    Recommend articles to a user based on the similarity between the new article and the user's previously liked articles.

    Args:
        user_likes (list[str]): A list of articles that the user has previously liked.
        new_article (str): The new article to be compared for recommendation.
        threshold (float, optional): The similarity score threshold for recommending an article. Defaults to 0.8.

    Returns:
        bool: Whether the article is recommended (True) or not (False).

    Raises:
        ValueError: If any of the articles are empty or not provided.
    """
    if not user_likes or not new_article:
        raise ValueError('Articles must not be empty or None.')

    tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/unsup-simcse-roberta-base')
    model = AutoModel.from_pretrained('princeton-nlp/unsup-simcse-roberta-base')

    # Tokenize and encode the articles as embeddings
    encoded_likes = tokenizer(user_likes, padding=True, return_tensors='pt')
    with torch.no_grad():
        user_likes_embeddings = model(**encoded_likes).pooler_output

    encoded_new_article = tokenizer([new_article], padding=True, return_tensors='pt')
    with torch.no_grad():
        new_article_embedding = model(**encoded_new_article).pooler_output

    # Calculate cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(new_article_embedding, user_likes_embeddings)
    max_similarity = torch.max(cos_sim).item()

    return max_similarity >= threshold

# test_function_code --------------------

def test_recommend_articles():
    print("Testing started.")
    user_likes = ['I love machine learning.', 'Data science is my passion.']
    new_article = 'Machine learning and data science are related fields.'

    # Test case 1: Recommend an article with high similarity
    print("Testing case [1/3] started.")
    assert recommend_articles(user_likes, new_article), f"Test case [1/3] failed: Expected recommendation for article with high similarity."

    # Test case 2: Do not recommend an article with low similarity
    new_article_low_sim = 'Cooking is an interesting hobby.'
    print("Testing case [2/3] started.")
    assert not recommend_articles(user_likes, new_article_low_sim), f"Test case [2/3] failed: Expected no recommendation for article with low similarity."

    # Test case 3: Raise ValueError for empty article
    empty_article = ''
    print("Testing case [3/3] started.")
    try:
        recommend_articles(user_likes, empty_article)
        assert False, f"Test case [3/3] failed: ValueError expected for empty article."
    except ValueError:
        pass
    print("Testing finished.")

# call_test_function_line --------------------

test_recommend_articles()
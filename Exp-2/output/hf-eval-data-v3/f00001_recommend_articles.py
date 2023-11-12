# function_import --------------------

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

# function_code --------------------

def recommend_articles(user_articles, new_articles, threshold=0.8):
    """
    Recommend articles to users based on how similar the articles are to their previously liked articles.
    
    Args:
        user_articles (list of str): List of user's previously liked articles.
        new_articles (list of str): List of new articles to be recommended.
        threshold (float, optional): The similarity threshold for recommending an article. Default is 0.8.
    
    Returns:
        recommended_articles (list of str): List of recommended articles.
    """
    tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/unsup-simcse-roberta-base')
    model = AutoModel.from_pretrained('princeton-nlp/unsup-simcse-roberta-base')
    
    user_embeddings = []
    for article in user_articles:
        inputs = tokenizer(article, return_tensors='pt')
        outputs = model(**inputs)
        user_embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
    user_embeddings = np.concatenate(user_embeddings, axis=0)
    
    recommended_articles = []
    for article in new_articles:
        inputs = tokenizer(article, return_tensors='pt')
        outputs = model(**inputs)
        new_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        similarities = cosine_similarity(new_embedding, user_embeddings)
        if np.max(similarities) >= threshold:
            recommended_articles.append(article)
    
    return recommended_articles

# test_function_code --------------------

def test_recommend_articles():
    user_articles = ['I love reading about technology.', 'Artificial Intelligence is fascinating.']
    new_articles = ['This new AI technology is groundbreaking.', 'Cooking is a form of art.', 'The world of Deep Learning.']
    recommended_articles = recommend_articles(user_articles, new_articles)
    assert 'This new AI technology is groundbreaking.' in recommended_articles
    assert 'The world of Deep Learning.' in recommended_articles
    assert 'Cooking is a form of art.' not in recommended_articles
    
    user_articles = ['I enjoy cooking.', 'Baking is therapeutic.']
    new_articles = ['The art of making sushi.', 'Understanding the stock market.', 'The science behind baking.']
    recommended_articles = recommend_articles(user_articles, new_articles)
    assert 'The art of making sushi.' in recommended_articles
    assert 'The science behind baking.' in recommended_articles
    assert 'Understanding the stock market.' not in recommended_articles
    
    return 'All Tests Passed'

# call_test_function_code --------------------

test_recommend_articles()
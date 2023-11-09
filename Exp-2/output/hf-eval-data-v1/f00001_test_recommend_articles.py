def test_recommend_articles():
    user_liked_articles = ['This is a great article about machine learning.', 'I love this article about natural language processing.']
    new_articles = ['This article about deep learning is very informative.', 'This is an interesting article about computer vision.', 'This article about NLP is very interesting.']
    recommended_articles = recommend_articles(user_liked_articles, new_articles)
    
    assert isinstance(recommended_articles, list), 'The result should be a list.'
    assert all(isinstance(article, str) for article in recommended_articles), 'All elements in the list should be strings.'
    assert 'This article about NLP is very interesting.' in recommended_articles, 'The NLP article should be recommended as it is similar to a liked article.'

test_recommend_articles()
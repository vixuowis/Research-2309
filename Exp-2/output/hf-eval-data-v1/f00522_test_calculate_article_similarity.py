def test_calculate_article_similarity():
    breaking_news_text = 'This is a breaking news article.'
    other_article_texts = ['This is another article.', 'This is yet another article.']
    similarities = calculate_article_similarity(breaking_news_text, other_article_texts)
    assert len(similarities) == len(other_article_texts), 'The number of similarity scores should be equal to the number of other articles.'
    assert all(0 <= score <= 1 for score in similarities), 'All similarity scores should be between 0 and 1.'

test_calculate_article_similarity()
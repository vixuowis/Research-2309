def test_analyze_customer_reviews():
    customer_reviews = ['I love this product!', 'This is the worst purchase I have ever made.', 'The product is okay, not great but not bad either.']
    seed_phrases = {
        'positive': ['This is great', 'I love it'],
        'neutral': ['It is okay', 'Not bad'],
        'negative': ['I hate it', 'This is terrible']
    }
    result = analyze_customer_reviews(customer_reviews, seed_phrases)
    assert 'positive' in result
    assert 'neutral' in result
    assert 'negative' in result
    assert 'I love this product!' in result['positive']
    assert 'The product is okay, not great but not bad either.' in result['neutral']
    assert 'This is the worst purchase I have ever made.' in result['negative']

test_analyze_customer_reviews()
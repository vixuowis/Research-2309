def test_extract_named_entities():
    """
    This function tests the 'extract_named_entities' function by using a sample news article.
    """
    news_article = 'Barack Obama visited the White House yesterday.'
    expected_output = ['Barack Obama', 'White House']
    assert set(extract_named_entities(news_article)) == set(expected_output), 'Test failed!'

    news_article = 'Apple Inc. is planning to open a new store in San Francisco.'
    expected_output = ['Apple Inc.', 'San Francisco']
    assert set(extract_named_entities(news_article)) == set(expected_output), 'Test failed!'

    print('All tests passed!')

test_extract_named_entities()
def test_classify_cat_or_dog():
    # Test the function with a cat image
    url = 'https://example.com/cat.jpg'
    assert classify_cat_or_dog(url) == 'cat'
    # Test the function with a dog image
    url = 'https://example.com/dog.jpg'
    assert classify_cat_or_dog(url) == 'dog'
def test_classify_image():
    """
    This function tests the classify_image function.
    It uses a sample image and text descriptions, and checks if the output is a list of probabilities.
    """
    # Define a sample image URL and text descriptions
    image_url = 'https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg'
    texts = ['文本描述1', '文本描述2', '文本描述3']
    
    # Call the classify_image function
    probs = classify_image(image_url, texts)
    
    # Check if the output is a list
    assert isinstance(probs, list), 'Output should be a list.'
    
    # Check if the length of the output list is equal to the number of text descriptions
    assert len(probs) == len(texts), 'Output list should have the same length as the number of text descriptions.'
    
    # Check if all elements in the output list are probabilities (between 0 and 1)
    assert all(0 <= prob <= 1 for prob in probs), 'All elements in the output list should be probabilities (between 0 and 1).'

test_classify_image()
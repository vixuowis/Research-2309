def test_classify_image():
    """
    This function tests the classify_image function.
    """
    # Define the image path and labels
    image_path = 'test_image.jpg'
    labels = ['label1', 'label2', 'label3']
    
    # Call the classify_image function
    probs = classify_image(image_path, labels)
    
    # Check the output
    assert isinstance(probs, list), 'The output should be a list.'
    assert len(probs) == len(labels), 'The output list should have the same length as the labels list.'
    for prob in probs:
        assert 0 <= prob <= 1, 'Each probability should be between 0 and 1.'

test_classify_image()
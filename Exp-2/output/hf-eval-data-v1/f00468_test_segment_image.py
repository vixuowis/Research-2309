def test_segment_image():
    '''
    This function tests the segment_image function by using a sample image from the Cityscapes dataset.
    '''
    # Define the URL of the sample image
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    # Call the segment_image function with the sample image
    logits = segment_image(url)

    # Assert that the output is not None
    assert logits is not None

    # Assert that the output has the correct shape
    assert logits.shape == (1, 19, 1024, 1024)

test_segment_image()
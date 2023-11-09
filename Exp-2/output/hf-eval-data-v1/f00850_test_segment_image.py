def test_segment_image():
    """
    This function tests the segment_image function by using a sample image.
    """
    image_path = 'sample_image.jpg'  # replace with the path to a sample image
    logits = segment_image(image_path)
    assert logits is not None, 'The output should not be None.'
    assert logits.shape[0] == 1, 'The output should have a batch size of 1.'
    assert logits.shape[1] == 19, 'The output should have 19 channels (for the 19 classes in the CityScapes dataset).'

test_segment_image()
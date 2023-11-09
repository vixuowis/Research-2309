def test_image_segmentation():
    """
    This function tests the image_segmentation function by segmenting a sample image and checking the output shape.
    """
    # Define a sample image URL
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    
    # Segment the image
    logits = image_segmentation(image_url)
    
    # Check the output shape
    assert logits.shape == (1, 150, 640, 640), 'The output shape is incorrect.'

test_image_segmentation()
def test_segment_aerial_image():
    """
    This function tests the 'segment_aerial_image' function.
    It uses an online image for the test.
    """
    # Define the URL of the test image
    test_image_url = 'https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg'
    
    # Download the test image
    response = requests.get(test_image_url, stream=True)
    response.raw.decode_content = True
    test_image = Image.open(response.raw)
    
    # Save the test image to a temporary file
    test_image_path = '/tmp/test_image.jpg'
    test_image.save(test_image_path)
    
    # Segment the test image
    predicted_semantic_map = segment_aerial_image(test_image_path)
    
    # Check the type of the result
    assert isinstance(predicted_semantic_map, type(Image.new('RGB', (1, 1)))), 'The result should be an image.'
    
    # Check the size of the result
    assert predicted_semantic_map.size == test_image.size, 'The result should have the same size as the input image.'

test_segment_aerial_image()
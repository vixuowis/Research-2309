def test_extract_captions():
    """
    This function tests the extract_captions function by comparing the output with the expected result.
    """
    # Test image URL
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    
    # Open the image
    with Image.open(requests.get(url, stream=True).raw) as image:
        # Generate captions
        preds = extract_captions(image)
    
    # Check if the output is a list
    assert isinstance(preds, list), 'Output should be a list.'
    
    # Check if the list is not empty
    assert len(preds) > 0, 'List should not be empty.'
    
    # Check if the elements of the list are strings
    assert all(isinstance(pred, str) for pred in preds), 'All elements of the list should be strings.'

test_extract_captions()
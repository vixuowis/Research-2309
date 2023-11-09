def test_get_image_embedding():
    """
    This function tests the get_image_embedding function.
    It loads a test image from the CortexBench dataset, passes it through the get_image_embedding function, and checks the output.
    """
    # Load a test image from the CortexBench dataset
    test_img = load_test_image_from_CortexBench()
    
    # Pass the test image through the get_image_embedding function
    test_embedding = get_image_embedding(test_img)
    
    # Check the output
    assert test_embedding is not None, 'The output embedding is None.'
    assert test_embedding.size() == (1, embd_size), 'The size of the output embedding is incorrect.'

test_get_image_embedding()
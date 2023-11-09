def test_classify_vehicle():
    """
    This function tests the classify_vehicle function.
    It uses an online image of a car for the test.
    """
    # The URL of the online image
    image_url = 'https://images.unsplash.com/photo-1541443131876-44b03de101c5'
    
    # Download the image
    response = requests.get(image_url)
    image_path = 'test_image.jpg'
    with open(image_path, 'wb') as file:
        file.write(response.content)
    
    # Classify the image
    probs = classify_vehicle(image_path)
    
    # Check the result
    assert isinstance(probs, torch.Tensor), 'The result should be a torch.Tensor.'
    assert probs.shape == (1, 4), 'The shape of the result should be (1, 4).'
    assert probs.sum().item() - 1 < 1e-6, 'The sum of the probabilities should be approximately 1.'
    
    # Delete the test image
    os.remove(image_path)

test_classify_vehicle()
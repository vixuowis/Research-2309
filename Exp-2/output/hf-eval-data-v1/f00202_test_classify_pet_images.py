def test_classify_pet_images():
    # Define the image path
    image_path = 'test_image.jpg'
    # Call the function with the test image
    probs = classify_pet_images(image_path)
    # Assert that the output is a PyTorch tensor
    assert isinstance(probs, torch.Tensor)
    # Assert that the output tensor has the correct shape
    assert probs.shape == (1, 2)
    # Assert that the sum of the probabilities is approximately 1
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

test_classify_pet_images()
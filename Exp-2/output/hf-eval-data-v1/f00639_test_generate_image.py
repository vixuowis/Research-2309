def test_generate_image():
    # Test the generate_image function
    
    # Define the test prompt
    test_prompt = 'astronaut playing guitar in space'
    
    # Generate the image
    test_image = generate_image(test_prompt)
    
    # Assert that an image was generated
    assert test_image is not None
    
    # Load the test dataset
    test_dataset = load_dataset('Stable Diffusion 1.5')
    
    # Select a sample from the dataset
    sample_prompt = test_dataset[0]['prompt']
    
    # Generate an image from the sample prompt
    sample_image = generate_image(sample_prompt)
    
    # Assert that an image was generated
    assert sample_image is not None

test_generate_image()
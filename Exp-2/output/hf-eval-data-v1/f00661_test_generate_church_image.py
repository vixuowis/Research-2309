def test_generate_church_image():
    # Call the function to generate church image
    generate_church_image()
    
    # Load the generated image
    image = Image.open('ddpm_generated_church_image.png')
    
    # Assert that the image is not None
    assert image is not None
    
    # Assert that the image mode is 'RGB'
    assert image.mode == 'RGB'
    
    # Assert that the image size is (256, 256)
    assert image.size == (256, 256)
import os

# Test function for generate_bedroom_image
# This function checks if the image file 'ddpm_generated_bedroom.png' is created in the current directory
def test_generate_bedroom_image():
    generate_bedroom_image()
    assert os.path.exists('ddpm_generated_bedroom.png'), 'Image file not found'
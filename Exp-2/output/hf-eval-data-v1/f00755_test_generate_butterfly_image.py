def test_generate_butterfly_image():
    """
    Test the generate_butterfly_image function.
    """
    generate_butterfly_image()
    assert os.path.exists('cute_butterfly_image.png'), 'Image not found'
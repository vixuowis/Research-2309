def test_generate_book_cover():
    '''
    This function tests the generate_book_cover function.
    It uses a sample image and prompt to generate a book cover.
    The function asserts that the output image file is created.
    
    Returns:
    None
    '''
    import os
    generate_book_cover('sample_image.png', 'A head full of roses', 'test_image_out.png')
    assert os.path.exists('test_image_out.png'), 'Output image not generated.'
    os.remove('test_image_out.png')
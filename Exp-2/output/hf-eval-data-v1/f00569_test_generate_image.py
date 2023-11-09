def test_generate_image():
    '''
    This function tests the generate_image function.
    '''
    prompt = 'A magical forest with unicorns and a rainbow.'
    output = generate_image(prompt)
    assert Path(output).is_file(), 'The generated image does not exist.'
    assert Path(output).suffix == '.png', 'The generated image is not a PNG file.'

test_generate_image()
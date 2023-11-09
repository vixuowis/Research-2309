def test_generate_painting():
    '''
    This function tests the generate_painting function.
    '''
    prompt = 'A head full of roses'
    image_url = 'https://huggingface.co/lllyasviel/control_v11p_sd15_normalbae/resolve/main/images/input.png'
    checkpoint = 'lllyasviel/control_v11p_sd15_normalbae'
    output = generate_painting(prompt, image_url, checkpoint)
    assert isinstance(output, str), 'The output should be a string.'
    assert output == 'images/image_out.png', 'The output image path is incorrect.'

test_generate_painting()
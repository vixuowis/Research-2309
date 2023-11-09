def test_process_image():
    '''
    This function tests the process_image function.
    '''
    import os
    input_image_path = 'test_input_image.png'
    output_image_path = 'test_output_image.png'
    num_inference_steps = 10
    process_image(input_image_path, output_image_path, num_inference_steps)
    assert os.path.exists(output_image_path), 'The processed image does not exist.'
    os.remove(output_image_path)
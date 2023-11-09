def test_generate_architectural_image():
    '''
    This function tests the generate_architectural_image function.
    '''
    image_path = "https://huggingface.co/lllyasviel/sd-controlnet-mlsd/resolve/main/images/room.png"
    generated_image_path = generate_architectural_image(image_path)
    assert isinstance(generated_image_path, str), "The function should return the path to the generated image as a string."
    assert generated_image_path.endswith('.png'), "The generated image should be a .png file."
    assert os.path.exists(generated_image_path), "The generated image file should exist."

test_generate_architectural_image()
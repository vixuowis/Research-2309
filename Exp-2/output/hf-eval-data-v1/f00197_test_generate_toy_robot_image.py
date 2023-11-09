def test_generate_toy_robot_image():
    '''
    This function tests the generate_toy_robot_image function.
    '''
    prompt = "toy robot"
    image_path = generate_toy_robot_image(prompt)
    assert os.path.exists(image_path), "The image was not generated."
    print("Test passed.")

test_generate_toy_robot_image()
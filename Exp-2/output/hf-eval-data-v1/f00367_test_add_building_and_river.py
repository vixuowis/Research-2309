def test_add_building_and_river():
    '''
    This function tests the add_building_and_river function by adding a building and a river to a sample landscape image.
    
    Returns:
    None
    '''
    import os
    from PIL import Image

    # Define the paths to the input and output images
    input_image_path = 'sample_landscape.jpg'
    output_image_path = 'output_image.png'

    # Call the function to add a building and a river to the landscape image
    add_building_and_river(input_image_path, output_image_path)

    # Check that the output image was saved correctly
    assert os.path.exists(output_image_path), 'The output image was not saved correctly.'

    # Load the output image
    output_image = Image.open(output_image_path)

    # Check that the output image has the correct mode (RGB)
    assert output_image.mode == 'RGB', 'The output image does not have the correct mode (RGB).'

    # Check that the output image has the correct size
    assert output_image.size == (512, 512), 'The output image does not have the correct size (512, 512).'
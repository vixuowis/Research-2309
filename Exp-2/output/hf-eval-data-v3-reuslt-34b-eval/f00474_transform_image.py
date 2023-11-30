# function_import --------------------

from transformers import pipeline

# function_code --------------------

def transform_image(input_image_path):
    """
    Transforms an input image using the 'GreeneryScenery/SheepsControlV5' model.

    Args:
        input_image_path (str): The path to the image file to be transformed.

    Returns:
        A stylized version of the input image.

    Raises:
        ValueError: If the model is not recognized by the Hugging Face library.
    """
    # Load model and set up function pipeline
    try:
        style_model = pipeline('image-style-transfer', 'GreeneryScenery/SheepsControlV5')
    except ValueError as error:
        raise Exception("An unknown model was received. Please use a model from the list of compatible models.")

    # Transform image
    output_image = style_model(input_image_path)
    
    return output_image

# test_function_code --------------------

def test_transform_image():
    """
    Tests the 'transform_image' function with a sample image.
    """
    sample_image_path = 'https://placekitten.com/200/300'
    try:
        transformed_image = transform_image(sample_image_path)
        assert transformed_image is not None
        print('Test Passed')
    except Exception as e:
        print('Test Failed: ', str(e))


# call_test_function_code --------------------

test_transform_image()
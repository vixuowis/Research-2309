# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_image_description(image):
    """
    Generate a description for an image using the 'microsoft/git-large-r-textcaps' model.

    Args:
        image (PIL.Image): The image to generate a description for.

    Returns:
        str: The generated description of the image.
    """
    description_generator = pipeline('text-generation', model='microsoft/git-large-r-textcaps')
    image_description = description_generator(image)
    return image_description

# test_function_code --------------------

def test_generate_image_description():
    """
    Test the 'generate_image_description' function.
    """
    # Load a test image
    test_image = Image.open('test_image.jpg')
    # Generate a description for the test image
    description = generate_image_description(test_image)
    # Assert that the description is a string
    assert isinstance(description, str)

# call_test_function_code --------------------

test_generate_image_description()
# function_import --------------------

from transformers import GenerativeImage2TextModel

# function_code --------------------

def generate_image_description(product_image):
    """
    Generate a product description based on the input image.

    Args:
        product_image (PIL.Image.Image): The product image.

    Returns:
        str: The generated product description.

    Raises:
        Exception: If the model fails to generate a description.
    """
    try:
        git_model = GenerativeImage2TextModel.from_pretrained('microsoft/git-large-coco')
        product_description = git_model.generate_image_description(product_image)
        return product_description
    except Exception as e:
        print(f'Failed to generate description: {e}')
        raise

# test_function_code --------------------

def test_generate_image_description():
    """
    Test the generate_image_description function.

    Raises:
        Exception: If the function fails to generate a description.
    """
    try:
        # Load a test image
        test_image = PIL.Image.open('test_image.jpg')
        description = generate_image_description(test_image)
        assert isinstance(description, str), 'The function should return a string.'
    except Exception as e:
        print(f'Test failed: {e}')
        raise

# call_test_function_code --------------------

test_generate_image_description()
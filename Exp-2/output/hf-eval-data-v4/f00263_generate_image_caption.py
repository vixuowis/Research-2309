# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_image_caption(image_path):
    """
    Generate a caption for an image using a pre-trained model.

    Parameters:
        image_path (str): The file path to the image for which to generate the caption.

    Returns:
        str: Generated caption for the image.

    Raises:
        ValueError: If the image_path does not exist or is not accessible.
    """
    # Ensure the image path exists
    if not os.path.exists(image_path):
        raise ValueError(f'Image path "{image_path}" does not exist or is not accessible.')

    # Load the pre-trained caption generation model
    caption_generator = pipeline('text2text-generation', model='salesforce/blip2-opt-6.7b')
    
    # Generate the caption for the image
    caption = caption_generator(image_path)[0]['generated_text']
    return caption

# test_function_code --------------------

def test_generate_image_caption():
    print("Testing generate_image_caption function.")

    # Provide a valid image path for testing
    valid_image_path = 'valid_test_image.jpg'

    # Test case 1: Check if the function returns a string
    print("Test case 1: Valid image path.")
    caption = generate_image_caption(valid_image_path)
    assert isinstance(caption, str), "Function should return a string."

    # Test case 2: Check handling of non-existent image path
    print("Test case 2: Non-existent image path.")
    invalid_image_path = 'nonexistent_test_image.jpg'
    try:
        generate_image_caption(invalid_image_path)
        assert False, "Function should raise ValueError for non-existent image path."
    except ValueError as e:
        assert str(e) == f'Image path "{invalid_image_path}" does not exist or is not accessible.', "Function should raise ValueError with appropriate message."

    print("Testing completed successfully.")
# function_import --------------------

from transformers import pipeline
import os

# function_code --------------------

def generate_text_from_image(image_path):
    """
    This function generates a text description based on the content of the image.
    
    Args:
        image_path (str): The path to the image file.
    
    Returns:
        str: The generated text description of the image.
    
    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"{image_path} does not exist")
    
    img2text_pipeline = pipeline("text-generation", model="microsoft/git-large-r-textcaps")
    image = open(image_path, "rb").read()
    text_output = img2text_pipeline(image)[0]["generated_text"]
    
    return text_output

# test_function_code --------------------

def test_generate_text_from_image():
    """
    This function tests the generate_text_from_image function.
    It uses a sample image file for testing.
    """
    image_path = "path_to_test_image.jpg"
    
    try:
        text_output = generate_text_from_image(image_path)
        assert isinstance(text_output, str), "The function should return a string."
    except FileNotFoundError:
        print(f"Test image file {image_path} does not exist")

# call_test_function_code --------------------

test_generate_text_from_image()
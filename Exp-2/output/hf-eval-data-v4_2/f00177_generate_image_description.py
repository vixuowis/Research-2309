# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_image_description(image_path):
    """Generates a text description for an image at the specified path using a pre-trained model.

    Args:
        image_path (str): The filepath to the image to be processed.

    Returns:
        str: The generated text description for the image.

    Raises:
        FileNotFoundError: If the image file does not exist at the specified path.
        Exception: If the image cannot be processed or there is an issue with the pipeline.
    """
    try:
        img2text_pipeline = pipeline("text-generation", model="microsoft/git-large-r-textcaps")
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        text_output = img2text_pipeline(image_data)[0]["generated_text"]
        return text_output
    except FileNotFoundError as fnf_error:
        raise fnf_error
    except Exception as e:
        raise e

# test_function_code --------------------

def test_generate_image_description():
    from pathlib import Path

    print("Testing started.")

    # Testing case 1: Valid image file
    print("Testing case [1/2] started.")
    valid_image_path = 'valid_image.jpg'  # Replace with a path to a valid image file
    assert Path(valid_image_path).is_file(), f"Test case [1/2] failed: File '{valid_image_path}' does not exist"

    # Testing case 2: Missing image file
    print("Testing case [2/2] started.")
    invalid_image_path = 'invalid_image.jpg'
    try:
        generate_image_description(invalid_image_path)
        assert False, f"Test case [2/2] failed: FileNotFoundError was not raised for missing file"
    except FileNotFoundError:
        pass

    print("Testing finished.")

# call_test_function_line --------------------

test_generate_image_description()
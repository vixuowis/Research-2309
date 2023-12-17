# requirements_file --------------------

!pip install -U huggingface-hub transformers torch pathlib

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def stylize_image(input_image_path):
    """
    Transform the style of an input image using a pre-trained Image-to-Image model.

    Args:
        input_image_path (str): The path to the image file to be processed.

    Returns:
        str: The path to the stylized image file.
    
    Raises:
        FileNotFoundError: If the input_image_path does not exist.
        OSError: If the transformed image could not be saved.
    """
    # Verify that the input image path exists
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image file not found: {input_image_path}")

    # Initialize the image-to-image transformer
    image_transformer = pipeline('image-to-image', model='GreeneryScenery/SheepsControlV5')

    # Apply the transformation to the input image
    stylized_image = image_transformer(input_image_path)

    # Save the stylized image to a file
    stylized_image_path = 'stylized_' + os.path.basename(input_image_path)
    with open(stylized_image_path, 'wb') as f:
        f.write(stylized_image)

    return stylized_image_path

# test_function_code --------------------

from pathlib import Path

def test_stylize_image():
    print("Testing started.")
    sample_image_path = 'sample_image.jpg'
    # Ensure a sample image file exists
    Path(sample_image_path).write_text('This is a sample image content')

    # Testing case 1: Valid image path
    print("Testing case [1/3] started.")
    assert Path(stylize_image(sample_image_path)).exists(), f"Test case [1/3] failed: Stylized image not found"

    # Testing case 2: Non-existing image path
    print("Testing case [2/3] started.")
    try:
        stylize_image('non_existent.jpg')
    except FileNotFoundError:
        assert True
    else:
        raise AssertionError("Test case [2/3] failed: No FileNotFoundError for non-existent image path")

    # Clean up
    Path(sample_image_path).unlink(missing_ok=True)
    print("Testing finished.")

# call_test_function_line --------------------

test_stylize_image()
# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import Pix2StructForConditionalGeneration
import PIL.Image

# function_code --------------------

def generate_description_from_chart(image_path: str) -> str:
    """
    Generate a textual description for a given chart image using the pretrained model.

    Args:
        image_path (str): The file path to the chart image.

    Returns:
        str: Generated textual description of the chart.

    Raises:
        FileNotFoundError: If the image file does not exist.
        RuntimeError: If the model fails to generate text.
    """
    model = Pix2StructForConditionalGeneration.from_pretrained('google/pix2struct-chartqa-base')
    image = PIL.Image.open(image_path)
    generated_text = model.generate_text(image)
    return generated_text

# test_function_code --------------------

def test_generate_description_from_chart():
    print("Testing started.")
    image_path = 'path_to_chart_image.jpg'  # This should be replaced with an actual image path

    # Testing with a valid image path
    print("Testing case [1/2] started.")
    description = generate_description_from_chart(image_path)
    assert isinstance(description, str), f"Test case [1/2] failed: Expected string output, got {type(description)}"

    # Testing with an invalid image path
    print("Testing case [2/2] started.")
    try:
        generate_description_from_chart('invalid_path.jpg')
        assert False, 'Test case [2/2] failed: FileNotFoundError not raised for invalid path'
    except FileNotFoundError:
        pass

    print("Testing finished.")

# call_test_function_line --------------------

test_generate_description_from_chart()
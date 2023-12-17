# requirements_file --------------------

!pip install -U transformers Pillow

# function_import --------------------

from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from PIL import Image

# function_code --------------------

def extract_data_table_from_plot(image_path, prompt):
    """
    Extract data table from plot image using DePlot.

    Args:
        image_path (str): The file path of the plot image.
        prompt (str): A text prompt to guide the data extraction.

    Returns:
        str: The extracted data table as text.
    """
    model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
    processor = Pix2StructProcessor.from_pretrained('google/deplot')

    image = Image.open(image_path)
    inputs = processor(images=image, text=prompt, return_tensors='pt')
    predictions = model.generate(**inputs, max_new_tokens=512)
    data_table = processor.decode(predictions[0], skip_special_tokens=True)

    return data_table

# test_function_code --------------------

def test_extract_data_table_from_plot():
    print("Testing started.")
    sample_image_path = 'plot_image_path.png'  # Replace with a valid image path

    # Test case 1: Typical prompt
    expected_output_1 = '...expected output...'  # Replace with expected output for test case 1
    print("Testing case [1/1] started.")
    output_1 = extract_data_table_from_plot(sample_image_path, 'Generate underlying data table of the figure below:')
    assert output_1 == expected_output_1, f"Test case [1/1] failed: Expected {expected_output_1}, got {output_1}"
    print("Testing finished.")
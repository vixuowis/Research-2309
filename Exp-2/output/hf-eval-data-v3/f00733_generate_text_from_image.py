# function_import --------------------

from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

# function_code --------------------

def generate_text_from_image(image_path):
    """
    Generate textual descriptions for images.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The generated textual description of the image.
    """
    model = Pix2StructForConditionalGeneration.from_pretrained('google/pix2struct-base')
    processor = Pix2StructProcessor.from_pretrained('google/pix2struct-base')
    inputs = processor(images=[image_path], return_tensors='pt')
    outputs = model.generate(**inputs)
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return generated_text

# test_function_code --------------------

def test_generate_text_from_image():
    """
    Test the function generate_text_from_image.
    """
    image_path = 'https://placekitten.com/200/300'
    generated_text = generate_text_from_image(image_path)
    assert isinstance(generated_text, str), 'The output should be a string.'
    assert len(generated_text) > 0, 'The output string should not be empty.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_text_from_image()
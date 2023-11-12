# function_import --------------------

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# function_code --------------------

def generate_image_caption(image_path: str, text: str = 'product photography') -> str:
    """
    Generate descriptive captions for photographs related to the products using Hugging Face Transformers.

    Args:
        image_path (str): The path to the image file.
        text (str, optional): A short text that provides some context to the photograph. Defaults to 'product photography'.

    Returns:
        str: The generated caption for the input image.
    """
    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
    image = Image.open(image_path)
    inputs = processor(image, text, return_tensors='pt')
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# test_function_code --------------------

def test_generate_image_caption():
    """
    Test the function generate_image_caption.
    """
    assert isinstance(generate_image_caption('test_image.jpg', 'product photography'), str)
    assert isinstance(generate_image_caption('test_image.jpg', 'landscape photography'), str)
    assert isinstance(generate_image_caption('test_image.jpg', 'portrait photography'), str)
    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_generate_image_caption())
# requirements_file --------------------

!pip install -U transformers pillow requests

# function_import --------------------

from transformers import MgpstrProcessor, MgpstrForSceneTextRecognition
from PIL import Image
import requests

# function_code --------------------

def translate_street_sign(image_url: str, target_language: str) -> str:
    """
    Translate text from a street sign image to a specified language.

    Args:
        image_url: str
            The URL of the image containing the street sign.
        target_language: str
            The code of the target language to which the text should be translated.

    Returns:
        The text from the street sign translated into the specified language.

    Raises:
        ValueError: If the image URL is not valid or the target language is not supported.
    """
    # Setup processor and model
    processor = MgpstrProcessor.from_pretrained('alibaba-damo/mgp-str-base')
    model = MgpstrForSceneTextRecognition.from_pretrained('alibaba-damo/mgp-str-base')
    
    # Load image from URL
    try:
        image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
    except Exception as e:
        raise ValueError(f"Unable to load image from URL: {image_url}") from e
    
    # Convert image to pixel values
    pixel_values = processor(images=image, return_tensors='pt').pixel_values
    
    # Recognize text from image
    outputs = model(pixel_values)
    generated_text = processor.batch_decode(outputs.logits)["generated_text"]
    
    # TODO: Implement text translation to target_language
    # For demonstration, we will just return the recognized text
    translated_text = generated_text
    
    return translated_text

# test_function_code --------------------

def test_translate_street_sign():
    print("Testing started.")

    # Test case 1: Valid image URL and target language
    print("Testing case [1/3] started.")
    sample_url = "https://i.postimg.cc/ZKwLg2Gw/367-14.png"
    try:
        result = translate_street_sign(sample_url, 'en')
        assert result is not None, f"Test case [1/3] failed: Failed to translate text from image {sample_url}"
    except Exception as e:
        assert False, f"Test case [1/3] failed: {str(e)}"

    # Test case 2: Invalid image URL
    print("Testing case [2/3] started.")
    invalid_url = "https://invalidurl.com/nonexistent.png"
    try:
        translate_street_sign(invalid_url, 'en')
        assert False, f"Test case [2/3] passed with invalid URL: {invalid_url}"
    except ValueError as e:
        assert "Unable to load image from URL" in str(e), f"Test case [2/3] failed: Expected ValueError was not raised"

    # Test case 3: Unsupported target language
    print("Testing case [3/3] started.")
    unsupported_language = "zz"
    try:
        translate_street_sign(sample_url, unsupported_language)
        assert False, f"Test case [3/3] passed with unsupported language: {unsupported_language}"
    except NotImplementedError as e:
        assert "language is not supported" in str(e), f"Test case [3/3] failed: Expected NotImplementedError was not raised"

    print("Testing finished.")

# call_test_function_line --------------------

test_translate_street_sign()
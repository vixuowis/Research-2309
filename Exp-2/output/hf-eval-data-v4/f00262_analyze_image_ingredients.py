# requirements_file --------------------

!pip install -U transformers PIL requests

# function_import --------------------

from transformers import BlipProcessor, Blip2ForConditionalGeneration
from PIL import Image
import requests

# function_code --------------------

def analyze_image_ingredients(img_url, question):
    """
    Analyze the image of a food item to identify its ingredients.

    Parameters:
    img_url (str): URL of the image to be analyzed.
    question (str): Question text asking about the ingredients in the image.

    Returns:
    str: Ingredients information as a textual response.
    """
    processor = BlipProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
    model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    inputs = processor(raw_image, question, return_tensors='pt')
    out = model.generate(**inputs)
    ingredient_info = processor.decode(out[0], skip_special_tokens=True)
    return ingredient_info

# test_function_code --------------------

def test_analyze_image_ingredients():
    print("Testing started.")
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    question = "What are the ingredients of this dish?"

    # Test case: Check if the function returns a string
    print("Testing case [1/1] started.")
    ingredient_info = analyze_image_ingredients(img_url, question)
    assert isinstance(ingredient_info, str), f"Test case failed: The function did not return a string."
    print("Testing finished.")
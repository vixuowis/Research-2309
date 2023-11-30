# function_import --------------------

from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import requests

# function_code --------------------

def identify_street_location(image_url: str, choices: list):
    """
    Identify the location of a street-level image using the Hugging Face Transformers' CLIPModel.

    Args:
        image_url (str): The URL of the street-level image.
        choices (list): A list of possible locations.

    Returns:
        str: The location with the highest probability.
    """
    
    # load pretrained CLIP model from Hugging Face hub -----
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", additional_special_tokens=[])
    
    # load image and tokenize input -----
    image = Image.open(requests.get(image_url, stream=True).raw)
    textualized_choices = [f"This is {c}." for c in choices]
    inputs = clip_tokenizer(text=textualized_choices, images=image, return_tensors="pt", padding=True)
    
    # run model and get probabilities -----
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()[0]
    
    # get most probable location -----
    best_choice_idx = int(probs.argmax())
    best_choice = choices[best_choice_idx]
    
    return (best_choice, probs[best_choice_idx])

# test_function_code --------------------

def test_identify_street_location():
    """
    Test the identify_street_location function.
    """
    image_url = 'https://placekitten.com/200/300'
    choices = ['San Jose', 'San Diego', 'Los Angeles', 'Las Vegas', 'San Francisco']

    location = identify_street_location(image_url, choices)
    assert location in choices

    image_url = 'https://placekitten.com/200/300'
    choices = ['New York', 'Chicago', 'Boston', 'Seattle', 'Austin']

    location = identify_street_location(image_url, choices)
    assert location in choices

    image_url = 'https://placekitten.com/200/300'
    choices = ['London', 'Paris', 'Berlin', 'Rome', 'Madrid']

    location = identify_street_location(image_url, choices)
    assert location in choices

    return 'All Tests Passed'


# call_test_function_code --------------------

test_identify_street_location()
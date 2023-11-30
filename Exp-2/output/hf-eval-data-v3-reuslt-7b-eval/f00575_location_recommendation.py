# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def location_recommendation(image_url: str, choices: list):
    """
    This function uses the StreetCLIP model to generate probabilities for various cities based on images from potential locations.
    It identifies possible locations for new stores.

    Args:
        image_url (str): The URL of the image of the potential location.
        choices (list): A list of city options to classify images.

    Returns:
        dict: A dictionary with city names as keys and their corresponding probabilities as values.
    """
    
    image = Image.open(requests.get(image_url, stream=True).raw)  # load image from url

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")   # set up clip model and tokenizer
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") 
    
    inputs = processor(text=choices, images=[image], return_tensors="pt", padding=True)   # preprocess image and text
    outputs = model(**inputs)  # run inference
    probabilities = outputs.logits_per_image.softmax(dim=-1).detach().numpy()  # get softmax probabilities for each city
    
    recommendations = dict(zip(choices, list(probabilities)))   # create dictionary of city recommendations
    
    return recommendations

# test_function_code --------------------

def test_location_recommendation():
    """
    This function tests the location_recommendation function with different test cases.
    """
    test_case_1 = ('https://placekitten.com/200/300', ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'])
    result_1 = location_recommendation(*test_case_1)
    assert isinstance(result_1, dict), 'The result should be a dictionary.'
    assert len(result_1) == len(test_case_1[1]), 'The number of cities in the result should be equal to the number of choices.'

    test_case_2 = ('https://placekitten.com/200/300', ['San Jose', 'San Diego', 'Los Angeles', 'Las Vegas', 'San Francisco'])
    result_2 = location_recommendation(*test_case_2)
    assert isinstance(result_2, dict), 'The result should be a dictionary.'
    assert len(result_2) == len(test_case_2[1]), 'The number of cities in the result should be equal to the number of choices.'

    return 'All Tests Passed'


# call_test_function_code --------------------

test_location_recommendation()
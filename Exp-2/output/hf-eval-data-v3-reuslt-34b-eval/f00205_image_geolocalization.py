# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def image_geolocalization(url: str, choices: list):
    """
    This function uses a pretrained CLIP model to identify the location of a given image.

    Args:
        url (str): The URL of the image to be geolocalized.
        choices (list): A list of possible choices for the location of the image.

    Returns:
        dict: A dictionary with the location choices as keys and their corresponding probabilities as values.
    """

    # get the image
    response = requests.get(url, stream=True)
    
    # open the image from the URL request
    img = Image.open(response.raw).convert("RGB")

    # define the processor and model for the CLIP task
    model_name = 'openai/clip-vit-base-patch32'
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name) 
    
    # prepare the inputs and labels based on our task
    inputs = processor(text=choices, images=img, return_tensors='pt')

    # get the predicted probabilities for each choice
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    
    # map the probabilities to a dictionary and normalize them
    preds = list(logits_per_image.numpy()[0])
    norm = [float(i)/sum(preds) for i in preds]
    
    # create a dictionary of choices mapped to their probabilities
    pred_dict = dict(zip(choices, norm))
    
    return pred_dict

# test_function_code --------------------

def test_image_geolocalization():
    url = 'https://placekitten.com/200/300'
    choices = ['San Jose', 'San Diego', 'Los Angeles', 'Las Vegas', 'San Francisco']
    result = image_geolocalization(url, choices)
    assert isinstance(result, dict)
    assert len(result) == len(choices)
    assert all(isinstance(choice, str) for choice in result.keys())
    assert all(isinstance(prob, float) for prob in result.values())
    assert abs(sum(result.values()) - 1) < 1e-6
    return 'All Tests Passed'


# call_test_function_code --------------------

test_image_geolocalization()
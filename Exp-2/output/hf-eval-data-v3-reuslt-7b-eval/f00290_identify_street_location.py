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
    
    model_name = "openai/clip-vit-base-patch32" # CLIPModel from Hugging Face Transformers.
    processor_name = "openai/clip-vit-base-patch32" # CLIPProcessor from Hugging Face Transformers.
    
    model = CLIPModel.from_pretrained(model_name) # Load the model.
    processor = CLIPProcessor.from_pretrained(processor_name) # Load the processor.
    
    image_url = str(image_url) # Convert the image URL to a string.
        
    response = requests.get(image_url, stream=True).raw
    input_image = Image.open(response)
    preprocessed_img = processor(images=input_image, return_tensors="pt").pixel_values # Preprocess the image for CLIP. 
    
    image_embedding = model.get_image_features(preprocessed_img).unsqueeze(0) # Create an embedding of the image.
    text_embeddings = processor.encode("street", return_tensors='pt').unsqueeze(0) # Create an embedding of "street".
    
    logits_per_image, logits_per_text = model(pixel_values=preprocessed_img, text=text_embeddings) # Obtain the probabilities for each choice. 

    probs_list = [] # Create an empty list to store the probabilities of each choice.

    for i in range(len(choices)):
        logits_per_choice = model(pixel_values=image_embedding, text=text_embeddings[:, [i]])[0]
        probs_list.append(logits_per_choice.softmax(dim=-1)[-1].detach().numpy())
    
    probabilities = list(zip(choices, probs_list)) # Zip the choices with their respective probabilities. 
    highest_probability = max(probs_list) # Get the choice with the highest probability from the list of probabilities. 


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
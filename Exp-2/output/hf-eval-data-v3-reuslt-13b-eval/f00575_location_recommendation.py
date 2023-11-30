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

    # Initialize the StreetCLIP model using CLIP's built-in pretrained weights for a 768 dimension image embedding.
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    # Create a new processor to process the images.
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Download the image from the URL and resize it to 256 x 192 px.
    response = requests.get(image_url)
    resized_image = Image.open(response.raw).resize((256, 192))
    
    # Process the image using the CLIP processor and get the embedding vector.
    image_features = clip_processor([resized_image], return_tensors="pt", padding=True)
    image_embedding = clip_model.get_image_features(**image_features).squeeze()
    
    # Get the text description for each city from the list of choices and process it using the CLIP processor.
    city_text_descriptions, probabilities = [], []
    for choice in choices:
        text_inputs = clip_processor(text=[choice], return_tensors="pt", padding=True)
        text_embedding = clip_model.get_text_features(**text_inputs).squeeze()
        
        # Calculate the cosine similarity between image and text embeddings to get their probabilities.
        probability = Image.cosine_similarity(image_embedding, text_embedding)
        city_text_descriptions.append(choice), probabilities.append(probability[0])
    
    # Create a dictionary with the city names as keys and their corresponding probabilities as values.
    return dict(zip(city_text_descriptions, probabilities))


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
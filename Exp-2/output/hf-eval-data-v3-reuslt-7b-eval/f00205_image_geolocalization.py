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
    
    # Get the image
    response = requests.get(url)
    img = Image.open(response.raw).convert("RGB")

    # Initialize pretrained CLIP model
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", feature_extractor_type='image')
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    # Process image
    inputs = processor(text=['a photo of'], images=img, return_tensors="pt", padding=True)
    pixels = inputs["pixel_values"].unsqueeze(0).to("cuda")
    
    # Get location choices embeddings
    with torch.no_grad():
        location_embeddings = model.get_image_features(pixels, return_tensors="pt").to("cuda")[:, 0]
        
    # Compute distances between each image and each choice (using cosine similarity)
    distances = torch.cosine_similarity(location_embeddings.unsqueeze(1), model.get_text_features(choices, return_tensors="pt").to("cuda"), dim=2).mean(dim=1)
    
    # Normalize distances and build dictionary of possible locations with their corresponding probabilities
    probs = softmax(distances/distances.sum())
    locs = {choices[i]: round(probs[i].item(), 4) for i in range(len(probs))}
    
    return locs


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
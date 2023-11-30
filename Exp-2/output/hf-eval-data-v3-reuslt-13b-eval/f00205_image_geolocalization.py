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

    # Load pre-trained model and tokenizer
    print('Loading CLIP model...')
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load image from URL and prepare it for the model
    print('Loading image...')
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(text=[], images=image, return_tensors='pt', padding=True)

    # Get image features from the pre-trained model and normalize them
    print('Getting image features...')
    outputs = clip_model.get_image_features(**inputs)
    normed_outputs = outputs / outputs.norm(dim=-1, keepdim=True).numpy() # Normalization is important for the cosine similarity calculation

    # Load choices features and normalize them as well
    print('Getting choices features...')
    text = [f'This is a picture of {choice}.' for choice in choices]
    inputs_text = processor(text=text, return_tensors='pt', padding=True)
    outputs_text = clip_model.get_text_features(**inputs_text)
    normed_outputs_text = outputs_text / outputs_text.norm(dim=-1, keepdim=True).numpy() # Normalization is important for the cosine similarity calculation

    # Calculate cosine similarity between image and choices features
    print('Calculating cosine similarities...')
    cos = (normed_outputs @ normed_outputs_text.T).diagonal()
    reshaped_cos = [float(i) for i in cos] # Reshape into a list with floats instead of PyTorch tensors

    return dict(zip(choices, reshaped_cos))


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
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel


def image_geolocalization(url: str, choices: list):
    '''
    This function uses a pretrained CLIP model to identify the location of a given image.
    
    Parameters:
    url (str): The URL of the image to be localized.
    choices (list): A list of possible locations for the image.
    
    Returns:
    list: A list of probabilities for each location.
    '''
    # Load the pretrained CLIP model 'geolocal/StreetCLIP', which is optimized for image geolocalization capabilities.
    model = CLIPModel.from_pretrained('geolocal/StreetCLIP')
    # Create a CLIP processor using the same 'geolocal/StreetCLIP' model, which will help us reformat the input data.
    processor = CLIPProcessor.from_pretrained('geolocal/StreetCLIP')
    # Retrieve the image from a URL or local file, and process it using PIL's Image.open() function.
    image = Image.open(requests.get(url, stream=True).raw)
    # Use the processor to convert the text choices and image into tensors, and pass these into the model.
    inputs = processor(text=choices, images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    # The model will then output the estimated probabilities for each location, which can help us determine the most likely match for the image.
    probs = logits_per_image.softmax(dim=1)
    return probs.tolist()
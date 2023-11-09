import requests
from PIL import Image
from io import BytesIO
from transformers import ViTFeatureExtractor, ViTForImageClassification


def is_adult(url):
    '''
    This function determines if a person in an image is an adult.
    It uses a pretrained model from Hugging Face Transformers to classify the age of the person.
    
    Parameters:
    url (str): The URL of the image.
    
    Returns:
    bool: True if the person is an adult, False otherwise.
    '''
    # Download the image from the URL
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    
    # Load the pretrained model and the feature extractor
    model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
    transforms = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')
    
    # Preprocess the image
    inputs = transforms(image, return_tensors='pt')
    
    # Classify the age of the person in the image
    output = model(**inputs)
    proba = output.logits.softmax(1)
    predicted_age_class = proba.argmax(1)
    
    # Determine if the person is an adult
    return predicted_age_class >= 18
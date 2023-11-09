from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests


def image_segmentation(image_url):
    """
    This function segments an image using the SegformerForSemanticSegmentation model from Hugging Face Transformers.
    
    Parameters:
    image_url (str): The URL of the image to be segmented.
    
    Returns:
    logits (torch.Tensor): The output logits from the segmentation model.
    """
    # Load the feature extractor and model
    feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b5-finetuned-ade-640-640')
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-ade-640-640')
    
    # Open the image
    image = Image.open(requests.get(image_url, stream=True).raw)
    
    # Extract features from the image
    inputs = feature_extractor(images=image, return_tensors='pt')
    
    # Pass the extracted features into the model
    outputs = model(**inputs)
    
    # Get the output logits
    logits = outputs.logits
    
    return logits
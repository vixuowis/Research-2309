from transformers import AutoFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests
import torch.nn as nn


def segment_clothes(image_url):
    """
    Function to segment and identify clothing items in an uploaded image.
    
    Args:
        image_url (str): URL of the image to be segmented.
    
    Returns:
        pred_seg (torch.Tensor): Predicted segmentation map.
    """
    # Load the feature extractor and model
    extractor = AutoFeatureExtractor.from_pretrained('mattmdjaga/segformer_b2_clothes')
    model = SegformerForSemanticSegmentation.from_pretrained('mattmdjaga/segformer_b2_clothes')
    
    # Load the image
    image = Image.open(requests.get(image_url, stream=True).raw)
    
    # Extract features from the image
    inputs = extractor(images=image, return_tensors='pt')
    
    # Pass the extracted features to the model
    outputs = model(**inputs)
    logits = outputs.logits.cpu()
    
    # Transform the output logits into a predicted segmentation map
    upsampled_logits = nn.functional.interpolate(logits, size=image.size[::-1], mode='bilinear', align_corners=False)
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    
    return pred_seg
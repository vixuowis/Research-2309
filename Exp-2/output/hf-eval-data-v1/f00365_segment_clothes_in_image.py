from transformers import AutoFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn


def segment_clothes_in_image(image_path):
    """
    This function segments clothes in an image using a pre-trained SegFormer model.
    
    Parameters:
    image_path (str): The path to the image file.
    
    Returns:
    None
    """
    # Load the feature extractor and model
    extractor = AutoFeatureExtractor.from_pretrained('mattmdjaga/segformer_b2_clothes')
    model = SegformerForSemanticSegmentation.from_pretrained('mattmdjaga/segformer_b2_clothes')
    
    # Open the image
    image = Image.open(image_path)
    
    # Extract the features and create the input tensors
    inputs = extractor(images=image, return_tensors='pt')
    
    # Perform the image segmentation
    outputs = model(**inputs)
    
    # Process the output
    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(logits, size=image.size[::-1], mode='bilinear', align_corners=False)
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    
    # Display the result
    plt.imshow(pred_seg)
    plt.show()
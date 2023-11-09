from transformers import pipeline
import torch
import torchvision


def classify_medical_image(image_path):
    """
    This function classifies a medical image into one of the following categories: X-ray, MRI scan, or CT scan.
    It uses the 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224' model from Hugging Face for zero-shot image classification.
    
    Args:
    image_path (str): The path to the medical image to be classified.
    
    Returns:
    str: The predicted class of the medical image.
    """
    # Load the image classification model
    clip = pipeline('zero-shot-image-classification', model='microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    
    # Specify the possible class names
    possible_class_names = ['X-ray', 'MRI scan', 'CT scan']
    
    # Execute the classifier on the image
    result = clip(image_path, possible_class_names)
    
    # Return the class with the highest probability
    return max(result, key=result.get)
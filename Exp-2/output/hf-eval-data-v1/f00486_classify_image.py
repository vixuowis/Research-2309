from transformers import pipeline


def classify_image(image_path):
    """
    This function classifies an image into one of the following categories: 'landscape', 'cityscape', 'beach', 'forest', 'animals'.
    It uses the 'laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg' model from Hugging Face for zero-shot image classification.
    
    Args:
    image_path (str): The path to the image file.
    
    Returns:
    str: The predicted category for the image.
    """
    # Create an image-classification model
    clip = pipeline('image-classification', model='laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg')
    
    # Classify the image
    result = clip(image_path, class_names=['landscape', 'cityscape', 'beach', 'forest', 'animals'])
    
    # Return the predicted category
    return result[0]['label']
from transformers import pipeline


def classify_pet_image(image_path):
    """
    This function classifies the given image into 'cat' or 'dog' using a pretrained CLIP model.
    
    Parameters:
    image_path (str): The path to the image file.
    
    Returns:
    str: The predicted class label ('cat' or 'dog').
    """
    # Create a pipeline for image classification using the pretrained CLIP model
    clip = pipeline('image-classification', model='laion/CLIP-convnext_base_w-laion2B-s13B-b82K')
    
    # Define the possible class labels
    pet_labels = ['cat', 'dog']
    
    # Use the pipeline to classify the image
    classification_result = clip(image_path, pet_labels)
    
    # Return the predicted class label
    return classification_result[0]['label']
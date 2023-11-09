from transformers import pipeline


def identify_animal(image_path):
    """
    This function identifies whether the animal in the image is a cat or a dog.
    It uses a pre-trained model from Hugging Face's transformers library.
    The model is a zero-shot learning model, which means it can classify images into categories it has not been explicitly trained on.
    The model used is 'laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft', which is pre-trained on a diverse set of images.
    
    Parameters:
    image_path (str): The path to the image file.
    
    Returns:
    str: The predicted class ('cat' or 'dog').
    """
    # Create the image classification pipeline
    image_classifier = pipeline('image-classification', model='laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft')
    
    # Classify the image
    result = image_classifier(image_path, ['cat', 'dog'])
    
    # Return the predicted class
    return result[0]['label']
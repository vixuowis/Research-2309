from transformers import pipeline


def classify_food_images(image_path, food_classes=['pizza', 'sushi', 'sandwich', 'salad', 'cake']):
    """
    This function classifies food images using a pre-trained model from Hugging Face Transformers.
    The model used is 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k' which is capable of zero-shot image classification.
    
    Parameters:
    image_path (str): The path to the image to be classified.
    food_classes (list, optional): A list of possible food classes. Defaults to ['pizza', 'sushi', 'sandwich', 'salad', 'cake'].
    
    Returns:
    dict: The classification results.
    """
    # Create an image classification model using the pretrained model
    image_classifier = pipeline('image-classification', model='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
    
    # Classify the image
    result = image_classifier(image_path, possible_class_names=food_classes)
    
    return result
from transformers import pipeline


def classify_plant(image_path: str, labels: list) -> str:
    """
    Function to classify the type of plant in the image using the pretrained model 'laion/CLIP-convnext_base_w-laion2B-s13B-b82K'.
    
    Parameters:
    image_path (str): Path to the image file.
    labels (list): List of possible class names.
    
    Returns:
    str: The most probable plant name.
    """
    # Create an image classification model using the pipeline function and specify the model 'laion/CLIP-convnext_base_w-laion2B-s13B-b82K'.
    clip = pipeline('image-classification', model='laion/CLIP-convnext_base_w-laion2B-s13B-b82K')
    
    # Classify the type of plant in the image
    plant_classifications = clip(image_path, labels)
    
    # Get the top plant
    top_plant = plant_classifications[0]['label']
    
    return top_plant
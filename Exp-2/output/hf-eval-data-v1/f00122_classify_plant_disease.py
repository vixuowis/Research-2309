import clip
from PIL import Image


def classify_plant_disease(image_path):
    """
    This function classifies the disease of a plant based on an image.
    It uses a pre-trained model from Hugging Face for zero-shot image classification.
    
    Args:
    image_path (str): The path to the image of the plant.
    
    Returns:
    dict: A dictionary with the classification results. The keys are the candidate labels and the values are the probabilities.
    """
    # Load the pre-trained model
    model, preprocess = clip.load('timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k')
    
    # Preprocess the image
    image = preprocess(Image.open(image_path))
    
    # Define the candidate labels
    candidate_labels = ['healthy', 'pest-infested', 'fungus-infected', 'nutrient-deficient']
    
    # Classify the image
    logits = model(image.unsqueeze(0)).logits
    probs = logits.softmax(dim=-1)
    
    # Prepare the classification results
    classification_results = {label: prob.item() for label, prob in zip(candidate_labels, probs.squeeze())}
    
    return classification_results
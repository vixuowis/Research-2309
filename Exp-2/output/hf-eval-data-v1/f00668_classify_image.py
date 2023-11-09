from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from PIL import Image


def classify_image(image_path, labels):
    """
    This function classifies an image using the ChineseCLIPModel.
    Args:
    image_path (str): The path to the image to be classified.
    labels (list): The labels to classify the image against.
    Returns:
    list: The probabilities of each label.
    """
    # Load the pre-trained model and processor
    model = ChineseCLIPModel.from_pretrained('OFA-Sys/chinese-clip-vit-large-patch14-336px')
    processor = ChineseCLIPProcessor.from_pretrained('OFA-Sys/chinese-clip-vit-large-patch14-336px')
    
    # Open the image
    image = Image.open(image_path)
    
    # Prepare the inputs for the model
    inputs = processor(images=image, text=labels, return_tensors='pt', padding=True)
    
    # Get the model's output
    outputs = model(**inputs)
    
    # Get the logits per image
    logits_per_image = outputs.logits_per_image
    
    # Get the probabilities
    probs = logits_per_image.softmax(dim=1)
    
    return probs.tolist()
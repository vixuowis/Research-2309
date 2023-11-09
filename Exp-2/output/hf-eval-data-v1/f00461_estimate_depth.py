from transformers import AutoModel
from torchvision.io import read_image


def estimate_depth(image_path):
    """
    This function estimates the depth of an image taken from a construction site.
    It uses a pre-trained model from Hugging Face Transformers.
    
    Parameters:
    image_path (str): The path to the image file.
    
    Returns:
    Tensor: The estimated depth of the image.
    """
    # Load the image data from a file
    image_input = read_image(image_path)
    
    # Load the pre-trained model
    depth_estimator = AutoModel.from_pretrained('sayakpaul/glpn-nyu-finetuned-diode-221228-072509')
    
    # Estimate the depth of the image
    predicted_depth = depth_estimator(image_input.unsqueeze(0))
    
    return predicted_depth
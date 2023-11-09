from transformers import AutoModel
from PIL import Image
import torch


def estimate_depth(image_path):
    """
    This function estimates the depth of elements in architectural designs from 2D images.
    It uses a pre-trained model from Hugging Face Transformers.
    
    Parameters:
    image_path (str): The path to the image file.
    
    Returns:
    depth_pred (torch.Tensor): The estimated depth of the elements in the image.
    """
    # Load the pre-trained model
    model = AutoModel.from_pretrained('sayakpaul/glpn-nyu-finetuned-diode-221116-104421')
    
    # Load the image
    image = Image.open(image_path)
    
    # Convert the image to a tensor
    tensor_image = torch.tensor(image).unsqueeze(0)
    
    # Estimate the depth of the elements in the image
    depth_pred = model(tensor_image)
    
    return depth_pred
import torch
from transformers import AutoModel
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import requests
from io import BytesIO


def estimate_depth(image_url):
    """
    This function estimates the depth of the environment using a monocular image.
    It uses a pre-trained model from Hugging Face Transformers.
    
    Parameters:
    image_url (str): The URL of the input image.
    
    Returns:
    depth_map (torch.Tensor): The estimated depth map.
    """
    # Initialize the model
    model = AutoModel.from_pretrained('sayakpaul/glpn-nyu-finetuned-diode-221122-082237')

    # Define the image transformations
    transforms = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the image from the URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    # Preprocess the image
    input_image = transforms(image).unsqueeze(0)

    # Compute the depth map
    with torch.no_grad():
        depth_map = model(input_image)

    return depth_map
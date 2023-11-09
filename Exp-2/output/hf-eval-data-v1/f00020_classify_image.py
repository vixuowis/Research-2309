from urllib.request import urlopen
from PIL import Image
import timm
import torch


def classify_image(img_url):
    """
    This function classifies the object within an image using the 'mobilenetv3_large_100.ra_in1k' model.
    
    Parameters:
    img_url (str): URL of the image to be classified.
    
    Returns:
    torch.Tensor: Tensor of class probabilities.
    """
    # Load the image
    img = Image.open(urlopen(img_url))
    
    # Load the model with pretrained weights
    model = timm.create_model('mobilenetv3_large_100.ra_in1k', pretrained=True)
    model = model.eval()
    
    # Resolve the model data configuration and create the appropriate input transforms
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    
    # Apply the transformations to the input image
    input_tensor = transforms(img).unsqueeze(0)
    
    # Pass the transformed image to the model for classification
    output = model(input_tensor)
    
    return output
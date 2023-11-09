from urllib.request import urlopen
from PIL import Image
import timm
import torch


def classify_image(img_url):
    """
    This function classifies an image from a URL into a thousand categories using a pretrained ConvNeXt model.
    
    Parameters:
    img_url (str): The URL of the image to be classified.
    
    Returns:
    torch.Tensor: The output of the model containing the probabilities for each of the 1,000 categories.
    """
    # Load an image from a URL
    img = Image.open(urlopen(img_url))
    
    # Load the pretrained ConvNeXt model
    model = timm.create_model('convnext_base.fb_in1k', pretrained=True)
    model = model.eval()
    
    # Get the data configuration for the model
    data_config = timm.data.resolve_model_data_config(model)
    
    # Create the necessary image transformations
    transforms = timm.data.create_transform(**data_config, is_training=False)
    
    # Perform the transformations on the image and add an extra dimension
    img = transforms(img).unsqueeze(0)
    
    # Make a prediction with the model
    output = model(img)
    
    return output
from urllib.request import urlopen
from PIL import Image
import timm
import torch


def classify_product_image(url):
    """
    This function classifies a product image into relevant categories using a pretrained MobileNet-v3 model.
    
    Args:
    url (str): URL of the product image.
    
    Returns:
    torch.Tensor: Output tensor representing category probabilities.
    """
    # Load the product image
    img = Image.open(urlopen(url))
    
    # Load the pretrained MobileNet-v3 model
    model = timm.create_model('mobilenetv3_large_100.ra_in1k', pretrained=True)
    model = model.eval()
    
    # Create the data transform required for the model
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    
    # Pass the transformed product image to the model
    output = model(transforms(img).unsqueeze(0))
    
    return output
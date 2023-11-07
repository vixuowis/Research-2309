from typing import *
from PIL import Image
import torchvision.transforms as T
import torch

def depth_estimator(image):
    # Parameters:
    #     image (PIL.Image): The input image
    # Returns:
    #     predictions (torch.Tensor): The depth predictions
    
    # Convert image to tensor
    image_tensor = T.ToTensor()(image)

    # Normalize image
    image_tensor = T.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])(image_tensor)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Load pre-trained depth estimation model
    model = torch.hub.load('intel-isl/MiDaS', 'MiDaS')

    # Set model to evaluation mode
    model.eval()

    # Generate depth predictions
    with torch.no_grad():
        predictions = model(image_tensor)

    return predictions

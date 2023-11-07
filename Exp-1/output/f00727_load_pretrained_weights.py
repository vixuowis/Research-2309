from typing import *
import timm

def load_pretrained_weights(model):
    """Load pretrained weights into the model.

    Args:
    - model (nn.Module): The model to load the weights into."""
    pretrained_model = timm.create_model("resnet50d", pretrained=True)
    model.load_state_dict(pretrained_model.state_dict())

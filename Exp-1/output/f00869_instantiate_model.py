from typing import *
from transformers import ConvNextConfig, UperNetConfig, UperNetForSemanticSegmentation

def instantiate_model():
    """Instantiate the model with the appropriate backbone.

    Args:
        None

    Returns:
        model (UperNetForSemanticSegmentation): The instantiated model
    """
    backbone_config = ConvNextConfig(out_features=["stage1", "stage2", "stage3", "stage4"])

    config = UperNetConfig(backbone_config=backbone_config)
    model = UperNetForSemanticSegmentation(config)
    return model

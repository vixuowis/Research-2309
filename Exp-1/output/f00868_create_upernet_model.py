from typing import *
from transformers import SwinConfig, UperNetConfig, UperNetForSemanticSegmentation

def create_upernet_model():
    """Create a UperNet model for semantic segmentation.

    Returns:
        UperNetForSemanticSegmentation: The UperNet model for semantic segmentation.
    """
    backbone_config = SwinConfig(out_features=["stage1", "stage2", "stage3", "stage4"])

    config = UperNetConfig(backbone_config=backbone_config)
    model = UperNetForSemanticSegmentation(config)

    return model

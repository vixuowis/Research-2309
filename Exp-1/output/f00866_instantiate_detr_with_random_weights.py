from typing import *
from detr import DetrConfig, DetrForObjectDetection

def instantiate_detr_with_random_weights():
    
    # Instantiate DETR with randomly initialized weights for backbone + Transformer
    
    # Returns:
    #     model (DetrForObjectDetection): The instantiated DETR model
    
    config = DetrConfig(use_pretrained_backbone=False)
    model = DetrForObjectDetection(config)
    return model

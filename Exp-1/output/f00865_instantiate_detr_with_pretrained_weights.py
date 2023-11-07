from typing import *
from transformers import DetrConfig, DetrForObjectDetection

def instantiate_detr_with_pretrained_weights():
    """Instantiate DETR with randomly initialized weights for Transformer, but pre-trained weights for backbone"""
    config = DetrConfig()
    model = DetrForObjectDetection(config)

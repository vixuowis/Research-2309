from typing import *
from transformers import DetrImageProcessor

def collate_fn(batch):
    # Custom collate function to batch images together
    # Args:
    #     batch: List of dictionaries containing image data
    # Returns:
    #     batch: Dictionary containing batched image data
    pixel_values = [item['pixel_values'] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors='pt')
    labels = [item['labels'] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch

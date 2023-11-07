from typing import *
import torch

def collate_fn(batch):
    # Pad images (which are now `pixel_values`) to the largest image in a batch, and create a corresponding `pixel_mask`
    # to indicate which pixels are real (1) and which are padding (0).
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch

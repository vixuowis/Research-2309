from typing import *
import torch
from transformers import ViTFeatureExtractor, ViTModel

model_checkpoint = "dandelin/vilt-b32-mlm"

def generate_embeddings(image_paths):
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_checkpoint)
    model = ViTModel.from_pretrained(model_checkpoint)
    embeddings = []
    
    # Generate embeddings for a list of image paths.
    # 
    # Args:
    #     image_paths (list): A list of image file paths.
    # 
    # Returns:
    #     embeddings (list): A list of image embeddings.
    for image_path in image_paths:
        image = Image.open(image_path)
        inputs = feature_extractor(images=image, return_tensors='pt')
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze())
    return embeddings

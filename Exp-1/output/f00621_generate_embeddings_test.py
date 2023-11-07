from f00621_generate_embeddings import *
import torch
from PIL import Image

model_checkpoint = "dandelin/vilt-b32-mlm"


# Test case 1
image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", "path/to/image3.jpg"]
embeddings = generate_embeddings(image_paths)
assert len(embeddings) == len(image_paths)


# Test case 2
image_paths = []
embeddings = generate_embeddings(image_paths)
assert len(embeddings) == 0


# Test case 3
image_paths = ["path/to/image1.jpg"]
embeddings = generate_embeddings(image_paths)
assert len(embeddings) == 1

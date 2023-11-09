from transformers import DetrForSegmentation, DetrFeatureExtractor
from PIL import Image
import torch

# Function to segment an image using the DETR model
# @param image_path: The path to the image to be segmented
# @return: The segmented image

def segment_image(image_path):
    # Load the image
    image = Image.open(image_path)

    # Load the feature extractor and model
    feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50-panoptic')
    model = DetrForSegmentation.from_pretrained('facebook/detr-resnet-50-panoptic')

    # Extract features from the image
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Get the model's output
    outputs = model(**inputs)

    # Process the output
    processed_sizes = torch.as_tensor(inputs['pixel_values'].shape[-2:]).unsqueeze(0)
    result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]

    return result
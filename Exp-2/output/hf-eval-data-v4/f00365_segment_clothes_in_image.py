# requirements_file --------------------

!pip install -U transformers PIL matplotlib torch

# function_import --------------------

from transformers import AutoFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

# function_code --------------------

def segment_clothes_in_image(image_path):
    """
    Segments clothes in the provided image using a SegFormer model.

    Parameters:
    - image_path (str): The file path or URL to the image to be processed.

    Returns:
    - PIL.Image: The segmented image with clothes outlined.
    """
    # Initialize feature extractor and segmentation model with pre-trained weights
    extractor = AutoFeatureExtractor.from_pretrained('mattmdjaga/segformer_b2_clothes')
    model = SegformerForSemanticSegmentation.from_pretrained('mattmdjaga/segformer_b2_clothes')
    
    # Open the image file
    image = Image.open(image_path)
    
    # Extract features and prepare the image for the model
    inputs = extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    
    # Process the output logits
    logits = outputs.logits.cpu()
    # Resize the output to match the original image size
    upsampled_logits = F.interpolate(logits, size=image.size[::-1], mode='bilinear', align_corners=False)
    
    # Get the most likely class per pixel
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    processed_image = Image.fromarray(pred_seg.numpy().astype('uint8'))
    
    # Return the segmented image
    return processed_image

# test_function_code --------------------

def test_segment_clothes_in_image():
    print("Testing started.")
    image_path = 'test_image.jpg'  # Replace with the path to a test image

    # Testing case 1: Check if the function returns an instance of PIL.Image
    print("Testing case [1/1] started.")
    result = segment_clothes_in_image(image_path)
    assert isinstance(result, Image.Image), f"Test case [1/1] failed: The result is not an image."

    print("Testing finished.")

# Run the test function
test_segment_clothes_in_image()
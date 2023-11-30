# function_import --------------------

from transformers import AutoFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn

# function_code --------------------

def segment_clothes(image_path):
    """
    This function segments clothes in an image using a pre-trained SegFormer model.

    Args:
        image_path (str): The path to the image file or a URL.

    Returns:
        A matplotlib figure showing the segmented image.
    """    
        
    # Download the image from a URL if it's not already saved locally.
    if "https" in image_path:
        response = requests.get(image_path)
        image_path = image_path.split("/")[-1]
        with open(image_path, 'wb') as f:
            f.write(response.content)

    # Load the trained model and feature extractor.
    model = SegformerForSemanticSegmentation.from_pretrained("nielsr/segformer-b3-512-ade")
    feature_extractor = AutoFeatureExtractor.from_pretrained("nielsr/segformer-b3-512-ade")
    
    # Preprocess the image using the feature extractor.
    inputs = feature_extractor(images=image_path, return_tensors="pt")

    # Predict the segmented image and plot it.
    outputs = model(**inputs)
    predicted_logits = outputs.logits[0]

    # Plot the predicted logits.
    plt.figure(figsize=(21, 9))

    plt.subplot(131)
    plt.imshow(Image.open(image_path).convert("RGB"))
    plt.title("Input image")

    plt.subplot(132)    
    plt.imshow(inputs["pixel_values"][0].cpu().numpy().transpose(1, 2, 0), interpolation="nearest")
    plt.title("Preprocessed input image")

    plt.subplot(133);
    predicted_image = outputs.logits[0].argmax(-1).squeeze().detach().cpu().numpy()
    predicted_image = Image.fromarray(predicted_image)
    predicted_image.putpalette([0, 0, 0, 255])  
    plt.imshow(predicted_image.convert("RGB"))
    plt.title("Predicted segmentation")

# test_function_code --------------------

def test_segment_clothes():
    """
    This function tests the segment_clothes function with a few test cases.
    """
    # Test case 1: An image of a person wearing clothes
    url1 = 'https://placekitten.com/200/300'
    result1 = segment_clothes(url1)
    assert isinstance(result1, type(plt)), 'Test Case 1 Failed'

    # Test case 2: Another image of a person wearing clothes
    url2 = 'https://placekitten.com/400/600'
    result2 = segment_clothes(url2)
    assert isinstance(result2, type(plt)), 'Test Case 2 Failed'

    # Test case 3: Yet another image of a person wearing clothes
    url3 = 'https://placekitten.com/800/1200'
    result3 = segment_clothes(url3)
    assert isinstance(result3, type(plt)), 'Test Case 3 Failed'

    return 'All Tests Passed'


# call_test_function_code --------------------

test_segment_clothes()
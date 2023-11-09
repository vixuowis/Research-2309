# function_import --------------------

from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from PIL import Image
import requests

# function_code --------------------

def segment_city_layout(image_path):
    """
    This function uses a pre-trained Segformer model to perform semantic segmentation on an image.
    The model has been fine-tuned on the CityScapes dataset, making it ideal for analyzing city layouts.

    Args:
        image_path (str): The path to the image file to be analyzed.

    Returns:
        Tensor: The output tensor containing the segmentation results.
    """
    # Load the feature extractor and model
    feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b5-finetuned-cityscapes-1024-1024')
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-cityscapes-1024-1024')

    # Load the image
    image = Image.open(image_path)

    # Prepare the inputs
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Compute the segmentation
    outputs = model(**inputs)

    return outputs.logits

# test_function_code --------------------

def test_segment_city_layout():
    """
    This function tests the segment_city_layout function by using a sample image from the CityScapes dataset.
    The function asserts that the output is a tensor, which indicates that the segmentation was successful.
    """
    # Define the path to the sample image
    image_path = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    # Call the function with the sample image
    output = segment_city_layout(image_path)

    # Assert that the output is a tensor
    assert isinstance(output, torch.Tensor), 'Output should be a tensor.'

# call_test_function_code --------------------

test_segment_city_layout()
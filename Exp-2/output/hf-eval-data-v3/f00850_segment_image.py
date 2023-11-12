# function_import --------------------

import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image

# function_code --------------------

def segment_image(image_path):
    """
    This function uses a pre-trained model to perform semantic segmentation on an image.
    The image is divided into segments, each representing a different object or part of the scene.
    The model used is 'nvidia/segformer-b5-finetuned-cityscapes-1024-1024', which is fine-tuned for semantic segmentation tasks on the CityScapes dataset.

    Args:
        image_path (str): The path to the image file.

    Returns:
        logits (torch.Tensor): The output of the model, representing the probability distribution over different classes for each pixel in the image.
    """
    feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b5-finetuned-cityscapes-1024-1024')
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-cityscapes-1024-1024')
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    return logits

# test_function_code --------------------

def test_segment_image():
    """
    This function tests the 'segment_image' function with a sample image.
    The test will pass if the function successfully performs semantic segmentation on the image and returns the expected output.
    """
    image_path = 'https://placekitten.com/200/300'
    output = segment_image(image_path)
    assert isinstance(output, torch.Tensor), 'Output is not a torch.Tensor'
    assert output.shape == (1, 19, 1024, 1024), 'Output shape is not as expected'
    print('All Tests Passed')

# call_test_function_code --------------------

test_segment_image()
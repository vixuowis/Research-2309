# function_import --------------------

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image

# function_code --------------------

def segment_image(image_path):
    """
    This function uses a pre-trained Segformer model to perform semantic segmentation on an image.
    The model has been fine-tuned on the CityScapes dataset.

    Args:
        image_path (str): The path to the image file.

    Returns:
        logits (torch.Tensor): The model's output logits, which represent the predicted segmentations.
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
    This function tests the segment_image function by using a sample image.
    The function asserts that the output is a torch.Tensor, which indicates that the function is working correctly.
    """
    image_path = 'sample_image.jpg'  # replace with the path to a sample image
    output = segment_image(image_path)
    assert isinstance(output, torch.Tensor), 'Output should be a torch.Tensor'

# call_test_function_code --------------------

test_segment_image()
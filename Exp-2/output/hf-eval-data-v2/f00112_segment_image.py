# function_import --------------------

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests

# function_code --------------------

def segment_image(image_url):
    """
    This function segments an image using the SegformerForSemanticSegmentation model from Hugging Face Transformers.

    Args:
        image_url (str): The URL of the image to be segmented.

    Returns:
        logits (torch.Tensor): The output logits from the segmentation model.
    """
    # Load the feature extractor and model
    feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b5-finetuned-ade-640-640')
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-ade-640-640')

    # Open the image
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Extract features from the image
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Pass the extracted features into the model
    outputs = model(**inputs)

    # Get the logits
    logits = outputs.logits

    return logits

# test_function_code --------------------

def test_segment_image():
    """
    This function tests the segment_image function by segmenting a sample image and checking the output type and shape.
    """
    # Define a sample image URL
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    # Segment the image
    logits = segment_image(image_url)

    # Check the output type
    assert isinstance(logits, torch.Tensor), 'Output type is not torch.Tensor'

    # Check the output shape
    assert logits.shape[0] == 1, 'Output shape is not correct'

# call_test_function_code --------------------

test_segment_image()
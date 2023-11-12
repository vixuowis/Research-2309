# function_import --------------------

from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests

# function_code --------------------

def segment_aerial_image(image_path):
    """
    This function segments an aerial image using a pre-trained model from Hugging Face Transformers.

    Args:
        image_path (str): The path to the aerial image to be segmented.

    Returns:
        predicted_semantic_map (dict): A dictionary containing the segmented regions of the image.
    """
    image = Image.open(image_path)
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained('facebook/maskformer-swin-base-ade')
    inputs = feature_extractor(images=image, return_tensors='pt')
    model = MaskFormerForInstanceSegmentation.from_pretrained('facebook/maskformer-swin-base-ade')
    outputs = model(**inputs)
    predicted_semantic_map = feature_extractor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    return predicted_semantic_map

# test_function_code --------------------

def test_segment_aerial_image():
    """
    This function tests the segment_aerial_image function by segmenting a sample aerial image.
    """
    image_path = 'https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg'
    predicted_semantic_map = segment_aerial_image(image_path)
    assert isinstance(predicted_semantic_map, dict), 'The output should be a dictionary.'
    assert 'masks' in predicted_semantic_map, 'The output dictionary should contain a key named masks.'
    assert 'labels' in predicted_semantic_map, 'The output dictionary should contain a key named labels.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_segment_aerial_image()
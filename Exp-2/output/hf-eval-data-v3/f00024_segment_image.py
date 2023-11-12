# function_import --------------------

from transformers import DetrForSegmentation, DetrFeatureExtractor
from PIL import Image
import torch

# function_code --------------------

def segment_image(image_path):
    """
    Segments an image using the pre-trained model 'facebook/detr-resnet-50-panoptic'.

    Args:
        image_path (str): The path to the image to be segmented.

    Returns:
        dict: The segmented output as a panoptic_seg_id.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    image = Image.open(image_path)
    feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50-panoptic')
    model = DetrForSegmentation.from_pretrained('facebook/detr-resnet-50-panoptic')
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    processed_sizes = torch.as_tensor(inputs['pixel_values'].shape[-2:]).unsqueeze(0)
    result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]
    return result

# test_function_code --------------------

def test_segment_image():
    """
    Tests the function segment_image.
    """
    try:
        # Test with a valid image path
        result = segment_image('valid_image.jpg')
        assert isinstance(result, dict), 'The result should be a dictionary.'

        # Test with an invalid image path
        try:
            result = segment_image('invalid_image.jpg')
        except FileNotFoundError:
            pass
        else:
            assert False, 'Expected a FileNotFoundError.'

        print('All tests passed.')
    except AssertionError as e:
        print(e)

# call_test_function_code --------------------

test_segment_image()
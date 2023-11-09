# function_import --------------------

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image

# function_code --------------------

def segment_aerial_image(image_path):
    """
    This function segments an aerial image into different categories such as streets, buildings, and trees using the OneFormerForUniversalSegmentation model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        segmentation_map (PIL.Image.Image): The segmented image.
    """
    image = Image.open(image_path)
    processor = OneFormerProcessor.from_pretrained('shi-labs/oneformer_ade20k_swin_large')
    model = OneFormerForUniversalSegmentation.from_pretrained('shi-labs/oneformer_ade20k_swin_large')

    segmentation_inputs = processor(images=image, task_inputs=['semantic'], return_tensors='pt')
    segmentation_outputs = model(**segmentation_inputs)
    segmentation_map = processor.post_process_semantic_segmentation(segmentation_outputs, target_sizes=[image.size[::-1]])[0]
    return segmentation_map

# test_function_code --------------------

def test_segment_aerial_image():
    """
    This function tests the segment_aerial_image function by segmenting a sample image and checking the output type.
    """
    segmentation_map = segment_aerial_image('sample_aerial_city_view.jpg')
    assert isinstance(segmentation_map, Image.Image), 'The output should be a PIL Image.'

# call_test_function_code --------------------

test_segment_aerial_image()
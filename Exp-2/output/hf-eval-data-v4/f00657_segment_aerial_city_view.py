# requirements_file --------------------

!pip install -U transformers Pillow

# function_import --------------------

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image


# function_code --------------------

def segment_aerial_city_view(image_path):
    """
    Segment streets, buildings, and trees in an aerial photograph of the city.

    Parameters:
    image_path (str): Path to the aerial image file.

    Returns:
    dict: A dictionary containing the semantic segmentation map.
    """
    image = Image.open(image_path)
    processor = OneFormerProcessor.from_pretrained('shi-labs/oneformer_ade20k_swin_large')
    model = OneFormerForUniversalSegmentation.from_pretrained('shi-labs/oneformer_ade20k_swin_large')

    segmentation_inputs = processor(images=image, task_inputs=['semantic'], return_tensors='pt')
    segmentation_outputs = model(**segmentation_inputs)
    segmentation_map = processor.post_process_semantic_segmentation(segmentation_outputs, target_sizes=[image.size[::-1]])
    return segmentation_map[0]

# test_function_code --------------------


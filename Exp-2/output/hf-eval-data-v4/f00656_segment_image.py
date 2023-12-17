# requirements_file --------------------

!pip install -U transformers PIL requests

# function_import --------------------

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import requests

# function_code --------------------

def segment_image(image_url, task='semantic'):
    """
    Segments an image using the OneFormer model trained for universal segmentation.

    Parameters:
        image_url (str): The URL of the image to segment.
        task (str): The segmentation task to perform. Options are 'semantic', 'instance', or 'panoptic'.

    Returns:
        dict: A dictionary containing the segmented image map.
    """
    image = Image.open(requests.get(image_url, stream=True).raw)
    processor = OneFormerProcessor.from_pretrained('shi-labs/oneformer_coco_swin_large')
    model = OneFormerForUniversalSegmentation.from_pretrained('shi-labs/oneformer_coco_swin_large')

    task_inputs = processor(images=image, task_inputs=[task], return_tensors='pt')
    outputs = model(**task_inputs)

    if task == 'semantic':
        segmented_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    elif task == 'instance':
        segmented_map = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    elif task == 'panoptic':
        segmented_map = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    else:
        raise ValueError('The task must be semantic, instance, or panoptic.')

    return segmented_map

# test_function_code --------------------

def test_segment_image():
    print("Testing segment_image function.")
    sample_url = 'https://huggingface.co/datasets/shi-labs/oneformer_demo/blob/main/coco.jpeg'

    # Test case 1: Semantic segmentation
    print("Testing case [1/3]: Semantic segmentation.")
    semantic_result = segment_image(sample_url, task='semantic')
    assert isinstance(semantic_result, dict), "Test case [1/3] failed: The result should be a dictionary."

    # Test case 2: Instance segmentation
    print("Testing case [2/3]: Instance segmentation.")
    instance_result = segment_image(sample_url, task='instance')
    assert isinstance(instance_result, dict), "Test case [2/3] failed: The result should be a dictionary."

    # Test case 3: Panoptic segmentation
    print("Testing case [3/3]: Panoptic segmentation.")
    panoptic_result = segment_image(sample_url, task='panoptic')
    assert isinstance(panoptic_result, dict), "Test case [3/3] failed: The result should be a dictionary."

    print("Testing completed.")

# Run the test function
test_segment_image()
# requirements_file --------------------

!pip install -U transformers pillow

# function_import --------------------

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image

# function_code --------------------

def perform_image_segmentation(image_path, task_type):
    """
    Segments an image using the OneFormer model for a specified task type.

    Args:
        image_path (str): The file path to the image to be segmented.
        task_type (str): The type of segmentation task, can be 'semantic', 'instance', or 'panoptic'.

    Returns:
        dict: A dictionary containing the segmented map. The key is the task type.

    Raises:
        ValueError: If an invalid task type is provided.
    """
    if task_type not in ['semantic', 'instance', 'panoptic']:
        raise ValueError('Invalid task type specified. Choose from semantic, instance, or panoptic.')

    image = Image.open(image_path)
    processor = OneFormerProcessor.from_pretrained('shi-labs/oneformer_ade20k_swin_tiny')
    model = OneFormerForUniversalSegmentation.from_pretrained('shi-labs/oneformer_ade20k_swin_tiny')

    inputs = processor(images=image, task_inputs=[task_type], return_tensors='pt')
    outputs = model(**inputs)

    if task_type == 'semantic':
        result = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    elif task_type == 'instance':
        result = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]['segmentation']
    else: # task_type == 'panoptic':
        result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]['segmentation']

    return {task_type: result}

# test_function_code --------------------

def test_perform_image_segmentation():
    print("Testing started.")
    sample_image_path = 'sample_image.jpg'  # Assuming this is the path to a sample image

    # Test case 1: Semantic segmentation
    print("Testing case [1/3] started.")
    semantic_result = perform_image_segmentation(sample_image_path, 'semantic')
    assert isinstance(semantic_result, dict) and 'semantic' in semantic_result, "Test case [1/3] failed: Semantic segmentation did not return the expected result."

    # Test case 2: Instance segmentation
    print("Testing case [2/3] started.")
    instance_result = perform_image_segmentation(sample_image_path, 'instance')
    assert isinstance(instance_result, dict) and 'instance' in instance_result, "Test case [2/3] failed: Instance segmentation did not return the expected result."

    # Test case 3: Panoptic segmentation
    print("Testing case [3/3] started.")
    panoptic_result = perform_image_segmentation(sample_image_path, 'panoptic')
    assert isinstance(panoptic_result, dict) and 'panoptic' in panoptic_result, "Test case [3/3] failed: Panoptic segmentation did not return the expected result."
    print("Testing finished.")

# call_test_function_line --------------------

test_perform_image_segmentation()
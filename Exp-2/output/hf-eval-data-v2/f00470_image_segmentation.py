# function_import --------------------

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image

# function_code --------------------

def image_segmentation(image_path: str, task: str):
    """
    Perform image segmentation using the OneFormer model.

    Args:
        image_path (str): The path to the image to be segmented.
        task (str): The type of segmentation task. Can be 'semantic', 'instance', or 'panoptic'.

    Returns:
        A segmented map of the image according to the specified task.
    """
    image = Image.open(image_path)
    processor = OneFormerProcessor.from_pretrained('shi-labs/oneformer_ade20k_swin_tiny')
    model = OneFormerForUniversalSegmentation.from_pretrained('shi-labs/oneformer_ade20k_swin_tiny')

    inputs = processor(images=image, task_inputs=[task], return_tensors='pt')
    outputs = model(**inputs)

    if task == 'semantic':
        return processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    elif task == 'instance':
        return processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]['segmentation']
    elif task == 'panoptic':
        return processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]['segmentation']

# test_function_code --------------------

def test_image_segmentation():
    """
    Test the image_segmentation function.
    """
    image_path = 'test_image.jpg'
    tasks = ['semantic', 'instance', 'panoptic']

    for task in tasks:
        result = image_segmentation(image_path, task)
        assert isinstance(result, type(None)) or isinstance(result, (np.ndarray, torch.Tensor))

# call_test_function_code --------------------

test_image_segmentation()
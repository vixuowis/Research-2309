# requirements_file --------------------

!pip install -U transformers PIL

# function_import --------------------

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image

# function_code --------------------

def perform_image_segmentation(image_path, task='semantic'):
    """
    Perform image segmentation on the provided image using the OneFormer model.

    Args:
        image_path (str): The file path to the image to be segmented.
        task (str): The type of segmentation task to perform. Options are 'semantic', 'instance', or 'panoptic'.

    Returns:
        dict: A dictionary containing the segmented map.
    """
    # Load the image using Pillow
    image = Image.open(image_path)

    # Initialize the processor and model
    processor = OneFormerProcessor.from_pretrained('shi-labs/oneformer_ade20k_swin_tiny')
    model = OneFormerForUniversalSegmentation.from_pretrained('shi-labs/oneformer_ade20k_swin_tiny')

    # Prepare the inputs for the model
    inputs = processor(images=image, task_inputs=[task], return_tensors='pt')
    outputs = model(**inputs)

    # Post-process the outputs to obtain the segmented map
    if task == 'semantic':
        segmented_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    elif task == 'instance':
        segmented_map = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]['segmentation']
    elif task == 'panoptic':
        segmented_map = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]['segmentation']
    else:
        raise ValueError(f'Invalid task type: {task}')

    return segmented_map

# test_function_code --------------------

def test_perform_image_segmentation():
    print("Testing perform_image_segmentation function.")

    # You would replace the image_path with a valid image path for testing.
    image_path = 'path_to_test_image.jpg'
    tasks = ['semantic', 'instance', 'panoptic']

    for task in tasks:
        print(f"Testing task {task}...")
        try:
            segmented_map = perform_image_segmentation(image_path, task)
            assert segmented_map is not None, f'The segmented_map should not be None for task {task}'
            print(f"Test case for task {task} succeeded.")
        except Exception as e:
            print(f"Test case for task {task} failed: {e}")

    print("All tests completed.")

# Run the test function
if __name__ == '__main__':
    test_perform_image_segmentation()
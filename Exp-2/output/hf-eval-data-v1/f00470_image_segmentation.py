from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image


def image_segmentation(image_path, task):
    '''
    This function performs image segmentation using the OneFormer model.
    It can handle diverse image segmentation tasks such as semantic, instance, and panoptic segmentation.
    
    Parameters:
    image_path (str): The path to the image to be segmented.
    task (str): The type of segmentation task. Can be 'semantic', 'instance', or 'panoptic'.
    
    Returns:
    predicted_map (dict): The segmented map.
    '''
    image = Image.open(image_path)
    processor = OneFormerProcessor.from_pretrained('shi-labs/oneformer_ade20k_swin_tiny')
    model = OneFormerForUniversalSegmentation.from_pretrained('shi-labs/oneformer_ade20k_swin_tiny')

    inputs = processor(images=image, task_inputs=[task], return_tensors='pt')
    outputs = model(**inputs)

    if task == 'semantic':
        predicted_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    elif task == 'instance':
        predicted_map = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]['segmentation']
    elif task == 'panoptic':
        predicted_map = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]['segmentation']

    return predicted_map
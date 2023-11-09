from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image


def image_segmentation(image_path):
    '''
    This function uses the OneFormerForUniversalSegmentation model from Hugging Face Transformers
    to perform image segmentation on the provided image.
    
    Args:
    image_path (str): The path to the image file.
    
    Returns:
    predicted_semantic_map: The segmented image map.
    '''
    # Load the image data from a file
    image = Image.open(image_path)
    
    # Load the pre-trained 'shi-labs/oneformer_coco_swin_large' model
    processor = OneFormerProcessor.from_pretrained('shi-labs/oneformer_coco_swin_large')
    model = OneFormerForUniversalSegmentation.from_pretrained('shi-labs/oneformer_coco_swin_large')
    
    # Prepare the image and task inputs for the desired segmentation task
    semantic_inputs = processor(images=image, task_inputs=['semantic'], return_tensors='pt')
    
    # Utilize the pre-trained model to process the image
    semantic_outputs = model(**semantic_inputs)
    
    # Post-process the output to get the segmented image map
    predicted_semantic_map = processor.post_process_semantic_segmentation(semantic_outputs, target_sizes=[image.size[::-1]])[0]
    
    return predicted_semantic_map
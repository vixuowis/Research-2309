from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image


def segment_aerial_image(image_path):
    """
    This function segments an aerial image into different categories such as streets, buildings, and trees using the OneFormerForUniversalSegmentation model.
    
    Parameters:
    image_path (str): The path to the image file.
    
    Returns:
    segmentation_map: The segmented map of the input image.
    """
    # Load the image
    image = Image.open(image_path)
    
    # Create an instance of the OneFormerProcessor with the ADE20k pre-trained model
    processor = OneFormerProcessor.from_pretrained('shi-labs/oneformer_ade20k_swin_large')
    
    # Create an instance of the OneFormerForUniversalSegmentation model
    model = OneFormerForUniversalSegmentation.from_pretrained('shi-labs/oneformer_ade20k_swin_large')
    
    # Process the input image
    segmentation_inputs = processor(images=image, task_inputs=['semantic'], return_tensors='pt')
    
    # Get the segmentation outputs
    segmentation_outputs = model(**segmentation_inputs)
    
    # Post process the segmentation outputs to get the segmentation map
    segmentation_map = processor.post_process_semantic_segmentation(segmentation_outputs, target_sizes=[image.size[::-1]])[0]
    
    return segmentation_map
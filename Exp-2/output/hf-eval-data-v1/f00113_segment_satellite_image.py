from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image


def segment_satellite_image(image_path):
    """
    This function segments a satellite image using a pre-trained model from Hugging Face Transformers.
    The model used is 'shi-labs/oneformer_coco_swin_large', which is trained on the COCO dataset for universal image segmentation tasks.
    
    Args:
    image_path (str): The path to the satellite image to be segmented.
    
    Returns:
    predicted_semantic_map: The segmented image.
    """
    # Load the image data from a file
    image = Image.open(image_path)
    
    # Load the pre-trained model and processor
    processor = OneFormerProcessor.from_pretrained('shi-labs/oneformer_coco_swin_large')
    model = OneFormerForUniversalSegmentation.from_pretrained('shi-labs/oneformer_coco_swin_large')
    
    # Prepare the inputs for the model
    semantic_inputs = processor(images=image, task_inputs=['semantic'], return_tensors='pt')
    
    # Run the model
    semantic_outputs = model(**semantic_inputs)
    
    # Post-process the outputs to get the segmented image
    predicted_semantic_map = processor.post_process_semantic_segmentation(semantic_outputs, target_sizes=[image.size[::-1]])[0]
    
    return predicted_semantic_map
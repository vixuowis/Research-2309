from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from PIL import Image


def segment_aerial_image(image_path):
    """
    This function segments an aerial image using a pre-trained MaskFormer model.
    The model is trained on the ADE20k dataset and is suited for semantic segmentation tasks.
    
    Parameters:
    image_path (str): The path to the aerial image to be segmented.
    
    Returns:
    predicted_semantic_map: The segmented image.
    """
    # Load the image data from a file
    image = Image.open(image_path)
    
    # Initialize the feature extractor
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained('facebook/maskformer-swin-base-ade')
    
    # Prepare the inputs for the model
    inputs = feature_extractor(images=image, return_tensors='pt')
    
    # Load the pre-trained model
    model = MaskFormerForInstanceSegmentation.from_pretrained('facebook/maskformer-swin-base-ade')
    
    # Analyze the image and segment it
    outputs = model(**inputs)
    
    # Post-process the segmentation
    predicted_semantic_map = feature_extractor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    
    return predicted_semantic_map
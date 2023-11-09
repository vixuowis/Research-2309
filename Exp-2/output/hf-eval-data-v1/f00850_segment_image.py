from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image


def segment_image(image_path):
    """
    This function uses a pre-trained Segformer model to perform semantic segmentation on an image.
    
    Args:
        image_path (str): The path to the image file.
    
    Returns:
        logits (torch.Tensor): The output of the Segformer model, representing the segmented image.
    """
    feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b5-finetuned-cityscapes-1024-1024')
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-cityscapes-1024-1024')
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    return logits
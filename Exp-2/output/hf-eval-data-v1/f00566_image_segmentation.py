from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests

# Function to perform image segmentation using MaskFormer model
# @param image_url: URL of the image to be segmented
# @return: Predicted panoptic map containing recognized objects and their boundaries
def image_segmentation(image_url):
    # Instantiate the feature_extractor using MaskFormerFeatureExtractor.from_pretrained() method with 'facebook/maskformer-swin-tiny-coco' model as the pretrained model.
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained('facebook/maskformer-swin-tiny-coco')
    # Instantiate the model using MaskFormerForInstanceSegmentation.from_pretrained() method which is trained on COCO panoptic segmentation.
    model = MaskFormerForInstanceSegmentation.from_pretrained('facebook/maskformer-swin-tiny-coco')
    # Open the image using the Image class from PIL and the Image.open() method.
    image = Image.open(requests.get(image_url, stream=True).raw)
    # Preprocess the image using the feature_extractor for the MaskFormer model.
    inputs = feature_extractor(images=image, return_tensors='pt')
    # Pass the preprocessed image tensors into the model to get the object detection results and segmentation masks.
    outputs = model(**inputs)
    # Process the outputs using the feature_extractor.post_process_panoptic_segmentation() method.
    result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    # Return the predicted panoptic map containing recognized objects and their boundaries.
    return result['segmentation']
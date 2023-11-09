from transformers import AutoFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests

# Function to segment clothes from an image
# @param image_url: URL or local path of the image
# @return: Segmented image

def segment_clothes(image_url):
    # Load the feature extractor
    extractor = AutoFeatureExtractor.from_pretrained('mattmdjaga/segformer_b2_clothes')
    # Load the pretrained model
    model = SegformerForSemanticSegmentation.from_pretrained('mattmdjaga/segformer_b2_clothes')
    # Load the image
    image = Image.open(requests.get(image_url, stream=True).raw)
    # Preprocess the image
    inputs = extractor(images=image, return_tensors='pt')
    # Get the segmentation output
    outputs = model(**inputs)
    # Get the logits
    logits = outputs.logits.cpu()
    return logits
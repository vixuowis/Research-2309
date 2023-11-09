from transformers import AutoFeatureExtractor, RegNetForImageClassification
import torch
from PIL import Image


def classify_inventory_image(image_path):
    """
    This function classifies the type of an image for an inventory using the pre-trained model 'zuppif/regnet-y-040'.
    
    Parameters:
    image_path (str): The path to the image file.
    
    Returns:
    str: The predicted label for the image.
    """
    image = Image.open(image_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained('zuppif/regnet-y-040')
    model = RegNetForImageClassification.from_pretrained('zuppif/regnet-y-040')
    inputs = feature_extractor(image, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]
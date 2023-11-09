from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
import torch
from datasets import load_dataset


def classify_product(image):
    '''
    This function classifies the product image using the pretrained model 'facebook/convnext-large-224'.
    Args:
    image : The product image to be classified.
    Returns:
    The predicted label of the product image.
    '''
    # Load the feature extractor and model
    feature_extractor = ConvNextFeatureExtractor.from_pretrained('facebook/convnext-large-224')
    model = ConvNextForImageClassification.from_pretrained('facebook/convnext-large-224')

    # Preprocess the image
    inputs = feature_extractor(image, return_tensors='pt')

    # Generate logits representing the probability of each object category
    with torch.no_grad():
        logits = model(**inputs).logits

    # Identify the most likely object class
    predicted_label = logits.argmax(-1).item()

    return model.config.id2label[predicted_label]
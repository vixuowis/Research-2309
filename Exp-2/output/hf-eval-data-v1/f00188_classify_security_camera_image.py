from transformers import AutoFeatureExtractor, RegNetForImageClassification
import torch
from datasets import load_dataset

# Function to classify images from a security camera
# Uses the pretrained RegNet model from Hugging Face Transformers
# Returns the predicted label for the image

def classify_security_camera_image(security_camera_image):
    # Load the pretrained RegNet model
    model = RegNetForImageClassification.from_pretrained('zuppif/regnet-y-040')
    # Load the feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained('zuppif/regnet-y-040')
    # Extract features from the image
    inputs = feature_extractor(security_camera_image, return_tensors='pt')
    # Pass the features through the model to get classification logits
    with torch.no_grad():
        logits = model(**inputs).logits
    # Find the predicted label by selecting the category with the highest logit value
    predicted_label = logits.argmax(-1).item()
    # Return the predicted label
    return model.config.id2label[predicted_label]
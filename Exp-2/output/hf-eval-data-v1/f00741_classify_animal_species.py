from transformers import AutoFeatureExtractor, RegNetForImageClassification
from PIL import Image
import torch

# Function to classify animal species based on their images
# It uses a pre-trained model called RegNet from Hugging Face Transformers
# The model is designed for image classification tasks

def classify_animal_species(animal_image_path):
    # Load the image of the animal species
    image = Image.open(animal_image_path)
    # Load the pre-trained feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained('zuppif/regnet-y-040')
    # Load the pre-trained RegNet model
    model = RegNetForImageClassification.from_pretrained('zuppif/regnet-y-040')
    # Preprocess the image using the pre-trained feature extractor
    inputs = feature_extractor(image, return_tensors='pt')
    # Pass the processed image into the model to obtain logits
    with torch.no_grad():
        logits = model(**inputs).logits
    # The category with the highest logits corresponds to the predicted species of the animal
    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image

# Function to classify device type from an image
# @param image_path: Path to the image file
# @return: Classified device type

def classify_device(image_path):
    # Load pre-trained model
    model = ViTForImageClassification.from_pretrained('lysandre/tiny-vit-random')
    # Load feature extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained('lysandre/tiny-vit-random')
    # Open the image file
    image = Image.open(image_path)
    # Preprocess the image
    input_image = feature_extractor(images=image, return_tensors='pt')
    # Get the output from the model
    output = model(**input_image)
    # Get the classified device type
    device_type = output.logits.argmax(dim=1).item()
    return device_type
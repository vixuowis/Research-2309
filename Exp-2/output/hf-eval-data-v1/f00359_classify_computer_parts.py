from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

def classify_computer_parts(user_uploaded_image_file_path):
    '''
    This function takes in the file path of an image uploaded by the user and returns the predicted label of the computer part in the image.
    It uses the Vision Transformer (ViT) model 'google/vit-base-patch16-224' from Hugging Face Transformers for image classification.
    '''
    # Create an instance of the ViTImageProcessor class
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    # Load the pre-trained Vision Transformer (ViT) model
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    # Load the user-uploaded image
    image = Image.open(user_uploaded_image_file_path)
    # Preprocess the image
    inputs = processor(images=image, return_tensors='pt')
    # Run the preprocessed image through the model
    outputs = model(**inputs)
    logits = outputs.logits
    # Find the predicted class index
    predicted_class_idx = logits.argmax(-1).item()
    # Get the human-readable predicted label
    predicted_label = model.config.id2label[predicted_class_idx]
    return predicted_label
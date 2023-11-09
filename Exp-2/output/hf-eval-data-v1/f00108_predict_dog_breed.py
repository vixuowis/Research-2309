from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

def predict_dog_breed(url):
    '''
    This function takes an image URL as input and predicts the breed of the dog in the image using the Vision Transformer (ViT) model pre-trained on ImageNet-21k.
    
    Parameters:
    url (str): The URL of the image.
    
    Returns:
    str: The predicted breed of the dog.
    '''
    # Load the image from the given URL
    image = Image.open(requests.get(url, stream=True).raw)
    
    # Initialize the ViTImageProcessor with the pre-trained model 'google/vit-base-patch16-224'
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
    # Initialize the ViTForImageClassification model with the pre-trained model 'google/vit-base-patch16-224'
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    
    # Preprocess the image using the processor and obtain the input tensor for the model
    inputs = processor(images=image, return_tensors='pt')
    
    # Pass the input tensor to the model and get the logits as output
    outputs = model(**inputs)
    logits = outputs.logits
    
    # Find the predicted class index by finding the index with the highest logit value
    predicted_class_idx = logits.argmax(-1).item()
    
    # Return the predicted class label for the dog breed
    return model.config.id2label[predicted_class_idx]
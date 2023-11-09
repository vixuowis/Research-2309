from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests

def classify_houseplant(url):
    '''
    Classify the type of houseplant in an image.

    Parameters:
    url (str): The URL of the image to classify.

    Returns:
    str: The predicted type of the houseplant.
    '''
    # Download the image from the given URL
    image = Image.open(requests.get(url, stream=True).raw)

    # Load the pre-trained MobileNet V1 model for image classification
    preprocessor = AutoImageProcessor.from_pretrained('google/mobilenet_v1_0.75_192')
    model = AutoModelForImageClassification.from_pretrained('google/mobilenet_v1_0.75_192')

    # Preprocess the input image
    inputs = preprocessor(images=image, return_tensors='pt')

    # Pass the preprocessed image to the model for classification
    outputs = model(**inputs)

    # Obtain the predicted class index
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    # Print the result as the name of the houseplant type
    return model.config.id2label[predicted_class_idx]
from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
import torch

# Function to classify dog breed from an image
# @param user_uploaded_image: The image uploaded by the user
# @return: The predicted dog breed

def classify_dog_breed(user_uploaded_image):
    # Load the feature extractor and model
    feature_extractor = ConvNextFeatureExtractor.from_pretrained('facebook/convnext-tiny-224')
    model = ConvNextForImageClassification.from_pretrained('facebook/convnext-tiny-224')

    # Process the image
    inputs = feature_extractor(user_uploaded_image, return_tensors='pt')

    # Get the logits for each class
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get the predicted label
    predicted_label = logits.argmax(-1).item()

    # Convert the label to a human-readable class name
    dog_breed = model.config.id2label[predicted_label]

    return dog_breed
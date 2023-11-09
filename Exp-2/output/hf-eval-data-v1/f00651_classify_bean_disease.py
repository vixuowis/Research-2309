from transformers import pipeline
import os

# Function to classify bean disease
# This function uses a pre-trained model from Hugging Face Transformers to classify images of bean leaves
# The model has been trained to detect diseases in bean crops
# The function takes as input the path to an image of a bean leaf and returns the predicted disease

def classify_bean_disease(image_path):
    # Check if the image file exists
    if not os.path.isfile(image_path):
        raise ValueError(f"Image file not found: {image_path}")
    
    # Create an image classification model
    classifier = pipeline('image-classification', model='fxmarty/resnet-tiny-beans')
    
    # Classify the image
    result = classifier(image_path)
    
    # Return the result
    return result
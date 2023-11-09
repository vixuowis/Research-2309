from transformers import pipeline
import torch

# This function is used to classify whether an image indicates diabetic retinopathy or not.
# It uses the 'martinezomg/vit-base-patch16-224-diabetic-retinopathy' model from Hugging Face Transformers.
# The model is a fine-tuned version of google/vit-base-patch16-224 on the None dataset.
# It is designed for image classification tasks, specifically for diabetic retinopathy detection.
# The function takes the path of the image as input and returns the classification result.
def classify_diabetic_retinopathy(image_path):
    # Create an image classifier by loading the model with the pipeline function
    image_classifier = pipeline('image-classification', 'martinezomg/vit-base-patch16-224-diabetic-retinopathy')
    # Use the image classifier to predict whether the given image indicates diabetic retinopathy
    result = image_classifier(image_path)
    return result
from transformers import pipeline
import torch

# Function to classify spoken digits
# This function uses the Hugging Face Transformers library to classify spoken digits.
# It uses the 'audio-classification' pipeline and the pre-trained 'MIT/ast-finetuned-speech-commands-v2' model.
# The function takes as input the path to an audio file and returns the predicted digit.
def classify_spoken_digit(audio_sample_path):
    # Define the classifier
    spoken_digit_classifier = pipeline('audio-classification', model='MIT/ast-finetuned-speech-commands-v2')
    # Use the classifier to predict the digit
    digit_prediction = spoken_digit_classifier(audio_sample_path)
    # Return the predicted digit
    return digit_prediction
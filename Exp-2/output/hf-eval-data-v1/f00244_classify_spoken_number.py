from transformers import pipeline
import torch

# Function to classify spoken numbers
# This function uses the Hugging Face Transformers library to create an audio classification model
# The model 'mazkooleg/0-9up-wavlm-base-plus-ft' is used, which is fine-tuned to recognize spoken numbers (0-9) in English, specifically focused on young children's voices
# The function takes an audio file path as input and returns the predicted number

def classify_spoken_number(audio_file_path):
    # Create the audio classification model
    spoken_number_classifier = pipeline('audio-classification', model='mazkooleg/0-9up-wavlm-base-plus-ft')
    # Use the model to predict the spoken number in the audio file
    prediction = spoken_number_classifier(audio_file_path)
    # Return the predicted number
    return prediction
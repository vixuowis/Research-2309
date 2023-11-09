from transformers import pipeline
import torch

# Function to detect spoken numbers in a phone call recording
# This function uses the 'mazkooleg/0-9up-data2vec-audio-base-960h-ft' model from Hugging Face Transformers
# The model is trained on spoken digit recognition tasks and can classify individual spoken digits from 0 to 9
# The function takes as input the path to an audio file and returns a list of detected digits

def detect_spoken_numbers(audio_file_path):
    # Load the model
    digit_classifier = pipeline('audio-classification', model='mazkooleg/0-9up-data2vec-audio-base-960h-ft')
    
    # Load the audio file
    audio_file = torch.load(audio_file_path)
    
    # Use the model to detect digits in the audio file
    digits_detected = digit_classifier(audio_file)
    
    return digits_detected
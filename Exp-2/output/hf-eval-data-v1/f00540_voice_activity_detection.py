import torch
from pyannote.audio import Model

# Function to perform voice activity detection
# This function uses the pre-trained model 'popcornell/pyannote-segmentation-chime6-mixer6' from Hugging Face Transformers
# The model is specifically designed to perform voice activity detection
# It is used to analyze audio recordings and detect segments where voice activity is present
# These segments are then used for transcriptions

def voice_activity_detection(audio):
    # Load the pre-trained model
    model = Model.from_pretrained('popcornell/pyannote-segmentation-chime6-mixer6')
    
    # Perform voice activity detection on the audio
    voice_activity = model(audio)
    
    return voice_activity
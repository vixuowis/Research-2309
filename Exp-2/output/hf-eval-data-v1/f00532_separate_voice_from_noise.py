from huggingface_hub import hf_hub_download
from asteroid.models import ConvTasNet
import torch

# Function to separate voice from noise
# This function uses the pre-trained ConvTasNet_Libri2Mix_sepclean_8k model from Hugging Face Transformers
# The model is specifically designed for separating speech from background noise
# The function takes an audio file as input and returns the separated voice and noise

def separate_voice_from_noise(audio_file):
    # Download the pre-trained model
    repo_id = 'JorisCos/ConvTasNet_Libri2Mix_sepclean_8k'
    model_path = hf_hub_download(repo_id=repo_id)
    
    # Load the model
    model = ConvTasNet.from_pretrained(model_path)
    
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_file)
    
    # Use the model to separate the voice from the noise
    voice, noise = model.separate(waveform)
    
    return voice, noise
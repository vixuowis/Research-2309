from transformers import AutoModelForAudioToAudio

# Function to apply noise suppression to audio
# This function uses the DCCRNet_Libri1Mix_enhsingle_16k model from Hugging Face Transformers
# The model is trained on the Libri1Mix dataset and is an audio-to-audio denoising model
# It takes as input a noisy audio and returns the denoised audio

def apply_noise_suppression(noisy_audio):
    # Load the pre-trained model
    model = AutoModelForAudioToAudio.from_pretrained('JorisCos/DCCRNet_Libri1Mix_enhsingle_16k')
    # Apply noise suppression
    denoised_audio = model(noisy_audio)
    return denoised_audio
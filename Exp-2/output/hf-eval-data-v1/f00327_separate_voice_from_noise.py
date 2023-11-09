from huggingface_hub import hf_hub_download
from asteroid import ConvTasNet
import soundfile as sf


def separate_voice_from_noise(audio_file_path):
    """
    This function separates voice from background noise in a given audio file using the ConvTasNet_Libri2Mix_sepclean_8k model.
    
    Parameters:
    audio_file_path (str): The path to the audio file.
    
    Returns:
    str: The path to the cleaned audio file.
    """
    # Download the model
    repo_id = "JorisCos/ConvTasNet_Libri2Mix_sepclean_8k"
    model_files = hf_hub_download(repo_id=repo_id)
    
    # Load the model
    model = ConvTasNet.from_pretrained(model_files)
    
    # Load the audio file
    mixture, sr = sf.read(audio_file_path)
    
    # Use the model to separate the voice from the noise
    estimates = model.separate(mixture)
    
    # Save the cleaned audio to a new file
    clean_audio_file_path = audio_file_path.replace('.wav', '_clean.wav')
    sf.write(clean_audio_file_path, estimates[0], sr)
    
    return clean_audio_file_path
import soundfile as sf
from asteroid import ConvTasNet
from huggingface_hub import hf_hub_download


def separate_speakers(audio_file):
    """
    This function separates speakers from a recorded audio using the ConvTasNet_Libri2Mix_sepclean_8k model from Hugging Face Transformers.
    
    Parameters:
    audio_file (str): The path to the audio file to be processed.
    
    Returns:
    numpy.ndarray: A 2D array where each row corresponds to a separated speaker.
    """
    # Download the pretrained model
    model_weights = hf_hub_download(repo_id='JorisCos/ConvTasNet_Libri2Mix_sepclean_8k', filename='model.pth')
    
    # Load the model
    model = ConvTasNet.from_pretrained(model_weights)
    
    # Read the audio file
    mixture_audio, sample_rate = sf.read(audio_file)
    
    # Separate the speakers
    est_sources = model.separate(mixture_audio)
    
    return est_sources
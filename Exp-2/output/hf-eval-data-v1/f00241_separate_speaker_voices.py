from asteroid.models import ConvTasNet


def separate_speaker_voices(wavs):
    """
    This function separates speaker voices from mixed sound using the pretrained model 'ConvTasNet_Libri3Mix_sepclean_8k'.
    
    Parameters:
    wavs (numpy array): The mixed audio recordings.
    
    Returns:
    numpy array: The separated speaker voices.
    """
    # Load the pretrained model
    model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri3Mix_sepclean_8k")
    
    # Use the model to separate the speaker voices
    separated_audio = model.separate(wavs)
    
    return separated_audio
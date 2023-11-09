from transformers import pipeline


def separate_audio_sources(audio_file_path):
    """
    This function separates music and vocals from an audio file using a pretrained model.
    
    Parameters:
    audio_file_path (str): The path to the audio file.
    
    Returns:
    dict: A dictionary containing the separated sources.
    """
    # Create a pipeline using the 'audio-source-separation' task, and initialize it with the model 'mpariente/DPRNNTasNet-ks2_WHAM_sepclean'.
    # This model is trained for separating sources in audio files.
    audio_separator = pipeline('audio-source-separation', model='mpariente/DPRNNTasNet-ks2_WHAM_sepclean')
    
    # Pass the audio file to the pipeline, and the model processes the file, separating the different sources (e.g., vocals and instruments) in the audio.
    separated_sources = audio_separator(audio_file_path)
    
    return separated_sources
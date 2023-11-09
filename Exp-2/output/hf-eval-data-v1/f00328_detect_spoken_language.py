from speechbrain.pretrained import EncoderClassifier, load_audio


def detect_spoken_language(conference_call_audio_file_path):
    """
    This function detects the language being spoken in an audio file.
    It uses the 'speechbrain/lang-id-voxlingua107-ecapa' model from Hugging Face Transformers, which is trained on 107 different languages.
    
    Args:
        conference_call_audio_file_path (str): The path to the audio file.
    
    Returns:
        str: The predicted language.
    """
    # Initialize the language identification model
    language_id = EncoderClassifier.from_hparams(source='speechbrain/lang-id-voxlingua107-ecapa', savedir='/tmp')
    
    # Load the audio samples from the call
    signal = load_audio(conference_call_audio_file_path)
    
    # Use the model's classify_batch method to process the audio samples and predict the spoken language
    prediction = language_id.classify_batch(signal)
    
    return prediction
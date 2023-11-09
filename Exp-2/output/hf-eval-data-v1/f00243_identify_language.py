from speechbrain.pretrained import EncoderClassifier
import torchaudio


def identify_language(audio_url):
    """
    This function identifies the language spoken in an audio file.
    It uses the 'speechbrain/lang-id-voxlingua107-ecapa' model from Hugging Face Transformers.
    The model is trained on the VoxLingua107 dataset using SpeechBrain.
    It covers 107 different languages.
    
    Parameters:
    audio_url (str): The URL of the audio file.
    
    Returns:
    str: The predicted language.
    """
    # Load the language identification model
    language_id = EncoderClassifier.from_hparams(source='speechbrain/lang-id-voxlingua107-ecapa', savedir='/tmp')
    
    # Load the audio file
    signal = language_id.load_audio(audio_url)
    
    # Perform language identification
    prediction = language_id.classify_batch(signal)
    
    return prediction
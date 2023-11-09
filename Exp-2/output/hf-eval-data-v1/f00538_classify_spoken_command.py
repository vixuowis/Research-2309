from transformers import pipeline
import torchaudio


def classify_spoken_command(audio_file_path):
    """
    This function classifies a spoken command into specific keywords using the 'superb/wav2vec2-base-superb-ks' model from Hugging Face Transformers.
    The model is specifically trained for keyword spotting to recognize pre-registered keywords in speech.
    
    Parameters:
    audio_file_path (str): The path to the audio file containing the spoken command.
    
    Returns:
    str: The classified keyword.
    """
    # Create an audio classification model with the specified model checkpoint
    audio_classifier = pipeline('audio-classification', model='superb/wav2vec2-base-superb-ks')
    
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_file_path)
    
    # Classify the spoken command into a specific keyword
    keyword = audio_classifier(waveform, top_k=1)
    
    return keyword
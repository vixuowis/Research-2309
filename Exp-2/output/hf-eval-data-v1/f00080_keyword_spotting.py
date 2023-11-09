from transformers import pipeline


def keyword_spotting(audio_file_path: str, top_k: int = 5):
    """
    This function uses the Hugging Face Transformers library to perform keyword spotting.
    It uses the 'superb/hubert-base-superb-ks' model which is trained to recognize user commands in spoken language.
    
    Parameters:
    audio_file_path (str): The path to the audio file to be processed.
    top_k (int): The number of top predictions to return. Default is 5.
    
    Returns:
    list: A list of detected keywords.
    """
    # Create an audio classification model
    keyword_spotter = pipeline('audio-classification', model='superb/hubert-base-superb-ks')
    
    # Use the model to detect keywords in the audio file
    detected_keywords = keyword_spotter(audio_file_path, top_k=top_k)
    
    return detected_keywords
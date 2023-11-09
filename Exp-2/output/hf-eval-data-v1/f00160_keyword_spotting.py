from transformers import pipeline


def keyword_spotting(audio_file_path: str, top_k: int = 5):
    """
    Function to determine the keyword spoken in a recorded audio file using Hugging Face Transformers.
    
    Parameters:
    audio_file_path (str): Path to the audio file.
    top_k (int): Number of top predictions to return. Default is 5.
    
    Returns:
    list: List of top_k keyword predictions.
    """
    # Load the pre-trained 'superb/hubert-base-superb-ks' model using the pipeline function.
    classifier = pipeline('audio-classification', model='superb/hubert-base-superb-ks')
    
    # Use the created classifier to process the recorded audio file.
    keyword_predictions = classifier(audio_file_path, top_k=top_k)
    
    return keyword_predictions
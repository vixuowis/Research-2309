from datasets import load_dataset
from transformers import pipeline


def emotion_recognition(file_path, top_k=5):
    '''
    This function uses the 'superb/wav2vec2-base-superb-er' pre-trained model from PyTorch Transformers to classify emotions based on a given audio file.
    The input audio file should have a 16kHz sampling rate.
    
    Args:
    file_path (str): The path to the audio file.
    top_k (int, optional): The number of top predictions to return. Defaults to 5.
    
    Returns:
    list: A list of the top_k predicted emotions and their corresponding scores.
    '''
    # Load the emotion recognition classifier
    classifier = pipeline('audio-classification', model='superb/wav2vec2-base-superb-er')
    
    # Classify the emotions in the audio file
    labels = classifier(file_path, top_k=top_k)
    
    return labels
from typing import *
import numpy as np
import librosa

def prepare_dataset(file_path):
    """
    Prepare the dataset for training.
    
    Parameters:
        file_path (str): Path to the audio file.
    
    Returns:
        numpy.ndarray: Normalized features of the audio file.
    """
    
    # Load audio file
    audio, sr = librosa.load(file_path)
    
    # Apply pre-processing steps
    preprocessed_audio = preprocess_audio(audio)
    
    # Extract features
    features = extract_features(preprocessed_audio)
    
    # Normalize features
    normalized_features = normalize_features(features)
    
    return normalized_features

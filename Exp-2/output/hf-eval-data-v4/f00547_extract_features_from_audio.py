# requirements_file --------------------

!pip install -U transformers librosa torch numpy

# function_import --------------------

from transformers import HubertModel, Wav2Vec2Processor
import torch
import librosa
import numpy as np

# function_code --------------------

def extract_features_from_audio(audio_path):
    """
    Extracts features from an audio sample using the Hubert-large-ll60k model.
    
    Parameters:
    audio_path (str): The file path to the audio sample in .wav format.
    
    Returns:
    np.ndarray: The extracted features from the audio sample.
    """

    # Load the pre-trained Hubert-large-ll60k model and the processor
    model = HubertModel.from_pretrained('facebook/hubert-large-ll60k')
    processor = Wav2Vec2Processor.from_pretrained('facebook/hubert-large-ll60k')
    
    # Load the audio file and resample to 16000 Hz
    audio, _ = librosa.load(audio_path, sr=16000)
    
    # Encode the audio into input format for the model
    inputs = processor(audio, return_tensors='pt', sampling_rate=16000)
    input_values = inputs.input_values

    # Disable gradient calculation
    with torch.no_grad():
        # Extract features from the audio sample
        logits = model(input_values).logits

    # Convert the features to numpy array
    features = logits.cpu().detach().numpy()

    return features.squeeze(0)

# test_function_code --------------------

def test_extract_features_from_audio():
    print("Testing started.")
    
    # Assuming the path to a valid .wav file for testing purposes
    test_audio_path = "path/to/test_audio.wav"

    # Testing extraction of features from audio sample
    print("Testing extraction of features started.")
    features = extract_features_from_audio(test_audio_path)
    
    # Test case 1: The features are not empty
    assert features.size != 0, "Test case failed: Extracted features are empty."

    # Test case 2: The features are numpy array
    assert isinstance(features, np.ndarray), "Test case failed: The features are not a numpy array."

    # Test case 3: The features are two-dimensional
    assert features.ndim == 2, "Test case failed: The features are not two-dimensional as expected."

    print("Testing finished.")

test_extract_features_from_audio()
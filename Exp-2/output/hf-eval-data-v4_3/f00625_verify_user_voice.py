# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch", "scipy"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoProcessor, AutoModelForAudioXVector
import torch

# function_code --------------------

def verify_user_voice(user_voice_sample_path: str, known_user_voice_embeddings: dict, user_id: str) -> bool:
    """
    Verifies if a given user's voice sample matches the known voice embedding of that user.
    
    Args:
        user_voice_sample_path (str): The file path to the user's voice sample audio file.
        known_user_voice_embeddings (dict): A dictionary containing the user's unique ID as keys and their voice embeddings as values.
        user_id (str): The unique identifier for the user whose voice we want to verify.

    Returns:
        bool: A boolean, True if the voice matches the known user voice embedding, False otherwise.

    Raises:
        FileNotFoundError: If the user voice sample file path does not exist.
        KeyError: If the user_id is not found in the known_user_voice_embeddings dictionary.
        ValueError: If the provided audio sample is not at a 16kHz sample rate as required by the model.
    """
    import os
    from scipy.io import wavfile

    # Check if the voice sample exists
    if not os.path.exists(user_voice_sample_path):
        raise FileNotFoundError("The specified voice sample file was not found.")
    
    # Check if the user_id exists in the known embeddings
    if user_id not in known_user_voice_embeddings:
        raise KeyError("The specified user_id does not exist in known voice embeddings.")
    
    # Load the user's voice sample and verify that it's sampled at 16kHz
    sample_rate, audio_data = wavfile.read(user_voice_sample_path)
    if sample_rate != 16000:
        raise ValueError("The voice sample must be sampled at 16kHz.")

    # Initialize the processor and model
    processor = AutoProcessor.from_pretrained("anton-l/wav2vec2-base-superb-sv")
    model = AutoModelForAudioXVector.from_pretrained("anton-l/wav2vec2-base-superb-sv")

    # Process the user's voice sample and get the embedding using the loaded model
    inputs = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).embeddings

    # Compare the embeddings with the known user voice embeddings to verify the identity
    is_match = embeddings == known_user_voice_embeddings[user_id]

    return is_match

# test_function_code --------------------

from unittest.mock import Mock

import pytest

def test_verify_user_voice():
    print("Testing started.")
    # Assume `load_dataset` is mocked to return predefined data
    load_dataset = Mock(return_value=(["user_voice_sample_path.wav"], {"user123": [0.1, 0.2, 0.3]}))
    dataset, known_voice_embeddings = load_dataset("user_voice_samples")
    
    sample_data = dataset[0]  # Get the user voice sample path
    sample_user_id = "user123"  # Known user ID for testing

    # Test case 1: User voice sample should match the known embedding
    print("Testing case [1/3] started.")
    assert verify_user_voice(sample_data, known_voice_embeddings, sample_user_id), "Test case [1/3] failed: The user voice sample should match the known embedding."

    # Test case 2: A non-existing user ID should raise a KeyError
    print("Testing case [2/3] started.")
    with pytest.raises(KeyError):
        verify_user_voice(sample_data, known_voice_embeddings, "unknown_user_id")

    # Test case 3: A non-existing file path should raise a FileNotFoundError
    print("Testing case [3/3] started.")
    with pytest.raises(FileNotFoundError):
        verify_user_voice("non_existing_path.wav", known_voice_embeddings, sample_user_id)
    
    print("Testing finished.")

# Run the test function
# test_verify_user_voice()

# call_test_function_line --------------------

# test_verify_user_voice()
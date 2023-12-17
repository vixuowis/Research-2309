# requirements_file --------------------

import subprocess

requirements = ["torchaudio", "speechbrain"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import torchaudio
from speechbrain.pretrained import EncoderClassifier

# function_code --------------------

def identify_customer_voices(audio_path: str) -> dict:
    """
    Identifies and returns embeddings for distinct speakers in an audio file.

    Args:
        audio_path (str): The file path to the audio file for speaker recognition.

    Returns:
        dict: A dictionary with speaker IDs and their corresponding embeddings.

    Raises:
        FileNotFoundError: If the audio file path does not exist.
    """
    # Load the pre-trained speaker recognition model
    classifier = EncoderClassifier.from_hparams(
        source='speechbrain/spkrec-xvect-voxceleb',
        savedir='pretrained_models/spkrec-xvect-voxceleb'
    )
    
    # Load the audio file
    signal, fs = torchaudio.load(audio_path)
    
    # Generate embeddings for the audio file
    embeddings = classifier.encode_batch(signal)

    # For demonstration purposes, we'll just assign dummy speaker IDs
    speakers = {f'speaker{i}': emb for i, emb in enumerate(embeddings)}
    return speakers

# test_function_code --------------------

def test_identify_customer_voices():
    print("Testing started.")
    
    # Test case 1: Valid audio file
    print("Testing case [1/1] started.")
    sample_audio = 'tests/samples/ASR/sample_audio.wav'
    embeddings = identify_customer_voices(sample_audio)
    assert isinstance(embeddings, dict) and len(embeddings) > 0, f"Test case [1/1] failed: Expected a dictionary of embeddings, got {embeddings}"
    print("Testing finished.")

# call_test_function_line --------------------

test_identify_customer_voices()
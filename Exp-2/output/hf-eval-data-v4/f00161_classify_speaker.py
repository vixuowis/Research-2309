# requirements_file --------------------

!pip install -U torchaudio speechbrain

# function_import --------------------

import torchaudio
from speechbrain.pretrained import EncoderClassifier

# function_code --------------------

def classify_speaker(audio_file_path):
    """
    Classify the speaker in the given audio file.

    Parameters:
    audio_file_path (str): Path to the audio file that needs to be classified.

    Returns:
    embeddings: The embedding vector of the speaker in the audio file.
    """
    # Load the pre-trained speaker verification model
    classifier = EncoderClassifier.from_hparams(source='speechbrain/spkrec-xvect-voxceleb', savedir='pretrained_models/spkrec-xvect-voxceleb')
    
    # Load the audio file
    signal, fs = torchaudio.load(audio_file_path)
    
    # Generate speaker embeddings
    embeddings = classifier.encode_batch(signal)
    return embeddings

# test_function_code --------------------

def test_classify_speaker():
    print("Testing classify_speaker function.")

    # Load a sample audio file
    sample_audio_file = 'tests/samples/ASR/sample_speaker_audio.wav'

    # Expected output shape
    expected_embedding_shape = (1, 512) # This may vary depending on the model

    # Test case 1: Check if the function returns an embedding vector
    print("Test case [1/1] started.")
    embeddings = classify_speaker(sample_audio_file)

    assert embeddings.shape == expected_embedding_shape, f"Test case [1/1] failed: Expected embedding shape {expected_embedding_shape}, got {embeddings.shape}"
    print("Test case [1/1] passed.")
    print("Testing finished.")

# Run the test function
test_classify_speaker()
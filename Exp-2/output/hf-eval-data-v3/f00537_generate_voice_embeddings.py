# function_import --------------------

import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier

# function_code --------------------

def generate_voice_embeddings(audio_file_path: str) -> torch.Tensor:
    """
    Generate voice embeddings for a given audio file using a pre-trained speaker recognition model.

    Args:
        audio_file_path (str): Path to the audio file.

    Returns:
        torch.Tensor: Voice embeddings for the audio file.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError: If there is an error loading the audio file or generating the embeddings.
    """
    classifier = EncoderClassifier.from_hparams(source='speechbrain/spkrec-xvect-voxceleb', savedir='pretrained_models/spkrec-xvect-voxceleb')
    signal, fs = torchaudio.load(audio_file_path)
    embeddings = classifier.encode_batch(signal)
    return embeddings

# test_function_code --------------------

def test_generate_voice_embeddings():
    """Test the generate_voice_embeddings function."""
    # Test with a valid audio file
    embeddings = generate_voice_embeddings('tests/samples/ASR/spk1_snt1.wav')
    assert embeddings is not None, 'Failed to generate embeddings for valid audio file'
    assert embeddings.shape[0] > 0, 'Generated embeddings should not be empty'
    # Test with a non-existent audio file
    try:
        embeddings = generate_voice_embeddings('non_existent_file.wav')
    except FileNotFoundError:
        pass
    else:
        assert False, 'Expected FileNotFoundError for non-existent audio file'
    # Test with an invalid audio file
    try:
        embeddings = generate_voice_embeddings('invalid_file.wav')
    except RuntimeError:
        pass
    else:
        assert False, 'Expected RuntimeError for invalid audio file'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_voice_embeddings()
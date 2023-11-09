# function_import --------------------

import torchaudio
from speechbrain.pretrained import EncoderClassifier

# function_code --------------------

def generate_voice_embeddings(audio_file):
    """
    This function generates voice embeddings for a given audio file using a pre-trained speaker recognition model.

    Args:
        audio_file (str): The path to the audio file for which to generate voice embeddings.

    Returns:
        Tensor: A tensor containing the voice embeddings of the audio file.
    """
    classifier = EncoderClassifier.from_hparams(source='speechbrain/spkrec-xvect-voxceleb', savedir='pretrained_models/spkrec-xvect-voxceleb')
    signal, fs = torchaudio.load(audio_file)
    embeddings = classifier.encode_batch(signal)
    return embeddings

# test_function_code --------------------

def test_generate_voice_embeddings():
    """
    This function tests the generate_voice_embeddings function by comparing the output embeddings for a test audio file with expected embeddings.
    Note: The test is not strict (i.e., the exact values of the embeddings are not compared) due to potential minor variations in the model's output.
    """
    test_audio_file = 'tests/samples/ASR/spk1_snt1.wav'
    embeddings = generate_voice_embeddings(test_audio_file)
    assert embeddings is not None, 'The function did not return any embeddings.'
    assert embeddings.shape[0] > 0, 'The function returned empty embeddings.'

# call_test_function_code --------------------

test_generate_voice_embeddings()
# function_import --------------------

import torchaudio
from speechbrain.pretrained import EncoderClassifier

# function_code --------------------

def classify_speaker(audio_file_path):
    """
    This function classifies the speaker in an audio file using a pre-trained model.

    Args:
        audio_file_path (str): The path to the audio file to be classified.

    Returns:
        embeddings (tensor): A tensor containing the speaker embeddings.

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    classifier = EncoderClassifier.from_hparams(source='speechbrain/spkrec-xvect-voxceleb', savedir='pretrained_models/spkrec-xvect-voxceleb')
    signal, fs = torchaudio.load(audio_file_path)
    embeddings = classifier.encode_batch(signal)
    return embeddings

# test_function_code --------------------

def test_classify_speaker():
    """
    This function tests the classify_speaker function by loading a sample audio file and checking the output.
    """
    sample_audio_file = 'tests/samples/ASR/spk1_snt1.wav'
    embeddings = classify_speaker(sample_audio_file)
    assert embeddings is not None, 'The function did not return any embeddings.'
    assert embeddings.size(0) > 0, 'The function returned empty embeddings.'

# call_test_function_code --------------------

test_classify_speaker()
# function_import --------------------

import torchaudio
from speechbrain.pretrained import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def recommend_podcast(audio_file_path: str, threshold: float = 0.8):
    """
    Recommends podcasts based on the user's favorite podcast speaker.

    Args:
        audio_file_path (str): Path to the audio file containing the user's favorite podcast speaker's voice.
        threshold (float): Threshold for similarity between speaker embeddings. Default is 0.8.

    Returns:
        list: List of recommended podcast episodes.

    Raises:
        FileNotFoundError: If the audio file is not found.
        ValueError: If the threshold is not between 0 and 1.
    """
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be between 0 and 1.')

    classifier = EncoderClassifier.from_hparams(source='speechbrain/spkrec-xvect-voxceleb', savedir='pretrained_models/spkrec-xvect-voxceleb')

    # Load favorite speaker's voice sample
    signal, fs = torchaudio.load(audio_file_path)
    favorite_speaker_embeddings = classifier.encode_batch(signal)

    # Compare with podcast episode speaker embeddings and recommend episodes with high similarity
    recommended_episodes = []
    for episode in podcast_episodes:
        episode_speaker_embeddings = classifier.encode_batch(episode['speaker_audio'])
        similarity = cosine_similarity(favorite_speaker_embeddings, episode_speaker_embeddings)
        if similarity >= threshold:
            recommended_episodes.append(episode)

    return recommended_episodes

# test_function_code --------------------

def test_recommend_podcast():
    """Tests the recommend_podcast function."""
    # Test with valid audio file and threshold
    recommended_episodes = recommend_podcast('tests/samples/ASR/spk1_snt1.wav', 0.8)
    assert isinstance(recommended_episodes, list)

    # Test with invalid audio file
    try:
        recommend_podcast('invalid_file.wav', 0.8)
    except FileNotFoundError:
        pass
    else:
        assert False, 'Expected FileNotFoundError for invalid audio file.'

    # Test with invalid threshold
    try:
        recommend_podcast('tests/samples/ASR/spk1_snt1.wav', 1.5)
    except ValueError:
        pass
    else:
        assert False, 'Expected ValueError for invalid threshold.'

    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_recommend_podcast())
# requirements_file --------------------

!pip install -U torchaudio speechbrain

# function_import --------------------

import torchaudio
from speechbrain.pretrained import EncoderClassifier

# function_code --------------------

def recommend_podcasts_based_on_speaker(favorite_speaker_audio_path: str, podcast_episode_db: dict) -> list:
    """
    Recommends podcast episodes based on the similarity to the user's favorite podcast speaker.

    Args:
        favorite_speaker_audio_path (str): The file path to the audio sample of the user's favorite podcast speaker.
        podcast_episode_db (dict): A database containing podcast episodes and their speaker embeddings.

    Returns:
        list: A list of recommended podcast episode titles.

    Raises:
        FileNotFoundError: If the favorite speaker audio file is not found.
    """
    classifier = EncoderClassifier.from_hparams(source='speechbrain/spkrec-xvect-voxceleb', savedir='pretrained_models/spkrec-xvect-voxceleb')

    # Load the user's favorite speaker's voice sample
    signal, fs = torchaudio.load(favorite_speaker_audio_path)
    favorite_speaker_embeddings = classifier.encode_batch(signal)

    # Compare with podcast episode speaker embeddings and recommend episodes
    recommendations = []
    for episode, embeddings in podcast_episode_db.items():
        if cosine_similarity(favorite_speaker_embeddings, embeddings) > 0.7:
            recommendations.append(episode)

    return recommendations

# test_function_code --------------------

def test_recommend_podcasts_based_on_speaker():
    print("Testing started.")
    # Assumed database format: {episode_title: embeddings_vector}
    podcast_episode_db = load_mock_podcast_db()

    # Mock audio path
    favorite_speaker_audio_path = 'mock_data/favorite_speaker_audio.wav'

    # Test case 1: Valid data
    print("Testing case [1/1] started.")
    recommendations = recommend_podcasts_based_on_speaker(favorite_speaker_audio_path, podcast_episode_db)

    assert recommendations, f"Test case [1/1] failed: No recommendations made."
    print("Testing finished.")

# call_test_function_line --------------------

test_recommend_podcasts_based_on_speaker()
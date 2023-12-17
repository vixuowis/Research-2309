# requirements_file --------------------

!pip install -U speechbrain torchaudio

# function_import --------------------

import torchaudio
from speechbrain.pretrained import EncoderClassifier
import os


# function_code --------------------

def recommend_podcasts(favorite_speaker_audio_path: str, podcast_episode_dir: str, similarity_threshold: float = 0.75) -> list:
    '''
    Recommend podcasts that have speakers with a similar voice to a user's favorite podcast speaker.

    Parameters:
        favorite_speaker_audio_path (str): Path to an audio file of the user's favorite podcast speaker.
        podcast_episode_dir (str): Directory containing podcast episodes with precomputed speaker embeddings.
        similarity_threshold (float): Threshold for considering two speakers' voices to be similar. Default is 0.75.

    Returns:
        list: Filenames of recommended podcast episodes.
    '''
    classifier = EncoderClassifier.from_hparams(source='speechbrain/spkrec-xvect-voxceleb', savedir='pretrained_models/spkrec-xvect-voxceleb')

    signal, fs = torchaudio.load(favorite_speaker_audio_path)
    favorite_speaker_embeddings = classifier.encode_batch(signal)

    recommended_episodes = []
    for episode_filename in os.listdir(podcast_episode_dir):
        episode_path = os.path.join(podcast_episode_dir, episode_filename)
        episode_signal, episode_fs = torchaudio.load(episode_path)
        episode_embeddings = classifier.encode_batch(episode_signal)

        # Here one should implement a similarity measure, e.g., cosine similarity
        # Assuming the function get_similarity() returns a similarity score
        similarity_score = get_similarity(favorite_speaker_embeddings, episode_embeddings)

        if similarity_score >= similarity_threshold:
            recommended_episodes.append(episode_filename)

    return recommended_episodes


# test_function_code --------------------

def test_recommend_podcasts():
    print("Testing recommend_podcasts function.")
    # Assume a directory 'test_podcast_episodes' and a sample 'favorite_speaker_audio_test.wav' are present
    recommendations = recommend_podcasts('favorite_speaker_audio_test.wav', 'test_podcast_episodes')

    # Test case: Check if the function returns a list
    assert isinstance(recommendations, list), "The function should return a list of recommendations."

    # Test case: Check if items in the list are strings (filenames)
    assert all(isinstance(item, str) for item in recommendations), "List items should be filenames as strings."

    # Test case: At least one recommendation, assuming existence of a similar speaker in the test episodes
    assert len(recommendations) > 0, "There should be at least one recommended episode."

    print("Testing completed successfully.")

# Run the test function
if __name__ == '__main__':
    test_recommend_podcasts()

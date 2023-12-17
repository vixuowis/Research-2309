# requirements_file --------------------

!pip install -U transformers asteroid

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def separate_audio_sources(audio_file_path):
    """
    This function separates vocals from a song using a pre-trained model.

    :param audio_file_path: Path to the audio file that needs to be processed.
    :return: A dictionary with keys 'vocals' and 'accompaniment' containing the separated audio sources.
    """
    # Initialize the source separation model
    source_separation = pipeline('audio-source-separation', model='Awais/Audio_Source_Separation')

    # Separate the audio sources
    separated_audio_sources = source_separation(audio_file_path)

    # Usually, the first source is vocals and the second is accompaniment for karaoke
    return {
        'vocals': separated_audio_sources[0],
        'accompaniment': separated_audio_sources[1]
    }

# test_function_code --------------------

def test_separate_audio_sources():
    print("Testing started.")

    # Test with a provided audio file path
    audio_file_path = 'path_to_audio_file.mp3'  # Should be replaced with a real audio file path

    # Call the function to test
    results = separate_audio_sources(audio_file_path)

    # Test cases: Check if both keys 'vocals' and 'accompaniment' are present in the results
    assert 'vocals' in results, "Test case failed: 'vocals' key is missing in the results"
    assert 'accompaniment' in results, "Test case failed: 'accompaniment' key is missing in the results"

    print("Testing finished.")
    return results

test_separate_audio_sources()
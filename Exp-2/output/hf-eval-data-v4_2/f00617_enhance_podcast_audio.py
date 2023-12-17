# requirements_file --------------------

!pip install -U transformers asteroid

# function_import --------------------

from transformers import AutoModelForAudioToAudio

# function_code --------------------

def enhance_podcast_audio(podcast_file_path, enhanced_file_path):
    """
    Enhance the audio quality of a podcast by reducing background noise.

    Args:
        podcast_file_path (str): The file path to the input podcast audio.
        enhanced_file_path (str): The file path where the enhanced audio will be saved.

    Returns:
        str: The file path to the enhanced audio file.

    Raises:
        FileNotFoundError: If podcast_file_path does not exist.
        Exception: If enhancement fails for other reasons.
    """
    import os
    if not os.path.exists(podcast_file_path):
        raise FileNotFoundError(f'{podcast_file_path} does not exist')

    # Load the pre-trained model
    audio_enhancer_model = AutoModelForAudioToAudio.from_pretrained('JorisCos/DCCRNet_Libri1Mix_enhsingle_16k')

    # Enhance the audio quality
    enhanced_audio = audio_enhancer_model.enhance_audio(podcast_file_path)

    # Save the enhanced audio to a new file
    enhanced_audio.export(enhanced_file_path, format='mp3')
    return enhanced_file_path

# test_function_code --------------------

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio

def test_enhance_podcast_audio():
    print("Testing started.")
    test_audio_path = 'sample_podcast.wav'  # It should be a real audio file path for actual testing.
    test_enhanced_path = 'enhanced_sample_podcast.mp3'

    # Testing case 1: Check if the file exists after enhancement
    print("Testing case [1/1] started.")
    result_path = enhance_podcast_audio(test_audio_path, test_enhanced_path)
    assert os.path.exists(result_path), f"Test case [1/1] failed: Enhanced file {result_path} does not exist."
    print("Testing finished.")

# Run the test function
# test_enhance_podcast_audio()

# call_test_function_line --------------------

# test_enhance_podcast_audio()
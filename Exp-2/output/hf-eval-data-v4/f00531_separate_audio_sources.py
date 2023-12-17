# requirements_file --------------------

!pip install -U speechbrain torchaudio

# function_import --------------------

from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

# function_code --------------------

def separate_audio_sources(input_audio_file):
    """
    This function separates the background music and vocal from an audio file using the pre-trained SepFormer model.

    Parameters:
        input_audio_file (str): The path to the input audio file to separate.

    Returns:
        Tuple[str, str]: Filenames of the separated audio files for background music and vocal.
    """
    model = separator.from_hparams(source='speechbrain/sepformer-wsj02mix', savedir='pretrained_models/sepformer-wsj02mix')
    est_sources = model.separate_file(path=input_audio_file)
    background_music_file = 'background_music.wav'
    vocal_file = 'vocal.wav'
    torchaudio.save(background_music_file, est_sources[:, :, 0].detach().cpu(), 8000)
    torchaudio.save(vocal_file, est_sources[:, :, 1].detach().cpu(), 8000)
    return background_music_file, vocal_file

# test_function_code --------------------

def test_separate_audio_sources():
    print("Testing separate_audio_sources function.")
    # Assuming specific audio file 'test_audio.wav' for testing
    input_audio_file = 'test_audio.wav'
    background_music_file, vocal_file = separate_audio_sources(input_audio_file)

    # Test case 1: Check if the returned file paths for background music and vocal are correct
    assert background_music_file == 'background_music.wav', "Test case failed: The background music file path is incorrect."
    assert vocal_file == 'vocal.wav', "Test case failed: The vocal file path is incorrect."

    # Add additional test cases as needed

    print("Testing finished.")

# Run the test function
test_separate_audio_sources()
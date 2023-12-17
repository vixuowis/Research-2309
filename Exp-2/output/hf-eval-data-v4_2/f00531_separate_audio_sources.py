# requirements_file --------------------

!pip install -U speechbrain torchaudio

# function_import --------------------

from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

# function_code --------------------

def separate_audio_sources(audio_file_path):
    """
    Separates the background music and vocal from an audio file using a pre-trained SepFormer model.

    Args:
        audio_file_path (str): The file path of the audio file to be processed.

    Returns:
        tuple: A tuple containing the file paths of the separated background music and vocal.

    Raises:
        FileNotFoundError: If the audio file does not exist at the specified path.
        Exception: If separation fails due to unexpected issues.
    """
    # Load the pre-trained SepFormer model
    model = separator.from_hparams(source='speechbrain/sepformer-wsj02mix', savedir='pretrained_models/sepformer-wsj02mix')
    # Perform separation
    est_sources = model.separate_file(path=audio_file_path)
    # Save the separated sources to new audio files
    background_music_path = 'background_music.wav'
    vocal_path = 'vocal.wav'
    torchaudio.save(background_music_path, est_sources[:, :, 0].detach().cpu(), 8000)
    torchaudio.save(vocal_path, est_sources[:, :, 1].detach().cpu(), 8000)
    return background_music_path, vocal_path

# test_function_code --------------------

def test_separate_audio_sources():
    print("Testing started.")

    # Assuming 'test_audio_file.wav' contains sample data for testing
    audio_file_path = 'test_audio_file.wav'

    # Testing case 1: Separate audio sources
    print("Testing case [1/1] started.")
    try:
        background_music_path, vocal_path = separate_audio_sources(audio_file_path)
        assert os.path.exists(background_music_path), f"Test case [1/1] failed: Background music file not found at {background_music_path}"
        assert os.path.exists(vocal_path), f"Test case [1/1] failed: Vocal file not found at {vocal_path}"
    except Exception as e:
        assert False, f"Test case [1/1] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_separate_audio_sources()
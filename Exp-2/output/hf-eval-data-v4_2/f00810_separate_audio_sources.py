# requirements_file --------------------

!pip install -U asteroid librosa soundfile

# function_import --------------------

import asteroid
from asteroid.models import ConvTasNet_Libri3Mix_sepclean_8k
import librosa
import soundfile as sf

# function_code --------------------

def separate_audio_sources(audio_path, output_dir):
    """
    Separate the speaker sources from the original audio file to filter the noise.

    Args:
        audio_path (str): The file path to the input audio to be processed.
        output_dir (str): The directory where the output audio files will be saved.

    Returns:
        list of str: A list containing the paths to the output audio files.

    Raises:
        IOError: If the audio_path does not exist or is inaccessible.
        OSError: If the output_dir cannot be created or accessed.
    """
    model = ConvTasNet_Libri3Mix_sepclean_8k()
    audio, _ = librosa.load(audio_path, sr=None, mono=False)

    sep_sources = model.separate(audio)
    output_paths = []
    for i, source in enumerate(sep_sources):
        output_path = os.path.join(output_dir, f'separated_source_{i}.wav')
        sf.write(output_path, source, 8000)
        output_paths.append(output_path)

    return output_paths

# test_function_code --------------------

def test_separate_audio_sources():
    print("Testing started.")

    # Assume the existence of an input audio file and an output directory
    input_audio = 'input_audio.wav'  # replace with a valid audio filename
    output_dir = 'outputs'  # replace with a valid directory

    # Run the actual function
    separated_files = separate_audio_sources(input_audio, output_dir)

    # Perform tests
    expected_num_sources = 3
    print("Testing case [1/1] started.")
    assert len(separated_files) == expected_num_sources, f"Test case [1/1] failed: Expected {expected_num_sources} separated files, got {len(separated_files)}"
    for filepath in separated_files:
        assert os.path.exists(filepath), f"Test case [1/1] failed: The file {filepath} does not exist"

    print("Testing finished.")

# call_test_function_line --------------------

test_separate_audio_sources()
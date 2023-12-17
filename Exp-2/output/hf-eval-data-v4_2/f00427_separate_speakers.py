# requirements_file --------------------

!pip install -U huggingface_hub soundfile asteroid

# function_import --------------------

from huggingface_hub import hf_hub_download
import soundfile as sf
from asteroid.models import ConvTasNet

# function_code --------------------


    """Separates speakers from an audio file using the ConvTasNet_Libri2Mix model.

    Args:
        audio_path (str): The file path to the input audio file.

    Returns:
        list: A list of audio data arrays, one for each separated speaker.

    Raises:
        FileNotFoundError: If the input audio file cannot be found.
        RuntimeError: If the model cannot be loaded properly.
    """
    # Download the pre-trained ConvTasNet_Libri2Mix_sepclean_8k model
    model_path = hf_hub_download(repo_id='JorisCos/ConvTasNet_Libri2Mix_sepclean_8k')
    # Load the model
    model = ConvTasNet.from_pretrained(model_path)

    # Read the audio file
    audio_data, sample_rate = sf.read(audio_path)
    # Ensure the audio file is mono and has a sample rate of 8kHz
    if audio_data.ndim > 1 or sample_rate != 8000:
        raise RuntimeError('The input audio must be mono and have a sample rate of 8kHz')

    # Separate the speakers using the model
    separated_audio = model.separate(audio_data)

    return separated_audio

# test_function_code --------------------


def test_separate_speakers():
    print("Testing started.")
    
    # Load a sample audio file
    audio_path = 'sample_audio.wav'
    sf.write(audio_path, np.random.rand(8000 * 10), 8000)  # Generate a dummy mono audio file with 10 seconds length

    # Test case 1: Successful speaker separation
    print("Testing case [1/1] started.")
    try:
        separated_audio = separate_speakers(audio_path)
        assert isinstance(separated_audio, list), "Output is not a list."
        assert all(isinstance(audio, np.ndarray) for audio in separated_audio), "Not all items in output are numpy arrays."
        print("Test case [1/1] passed.")
    except Exception as e:
        print(f"Test case [1/1] failed: {e}")
    print("Testing finished.")

# call_test_function_line --------------------

test_separate_speakers()
# requirements_file --------------------

!pip install -U speechbrain torchaudio

# function_import --------------------

import os
import torchaudio
from speechbrain.pretrained import Tacotron2, HIFIGAN

# function_code --------------------

def generate_voiceover(text, output_dir):
    """
    Generates a voiceover from the input text using pre-trained Tacotron2 and HiFIGAN models.

    Args:
        text (str): Input text to be converted into speech.
        output_dir (str): Directory to save the generated voiceover.wav file.

    Returns:
        str: The file path of the generated voiceover .wav file.

    Raises:
        FileNotFoundError: If the output directory does not exist.
    """
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f'The directory {output_dir} does not exist')

    # Paths to store temporary model data
    tmpdir_tts = os.path.join(output_dir, 'tts')
    tmpdir_vocoder = os.path.join(output_dir, 'vocoder')

    # Initialize pre-trained models
    tacotron2 = Tacotron2.from_hparams(source='padmalcom/tts-tacotron2-german', savedir=tmpdir_tts)
    hifi_gan = HIFIGAN.from_hparams(source='padmalcom/tts-hifigan-german', savedir=tmpdir_vocoder)

    # Convert text to spectrogram
    mel_output, mel_length, alignment = tacotron2.encode_text(text)

    # Convert spectrogram to waveform
    waveforms = hifi_gan.decode_batch(mel_output)

    # Save waveform as a .wav file
    audio_file_path = os.path.join(output_dir, 'voiceover.wav')
    torchaudio.save(audio_file_path, waveforms.squeeze(1), 22050)

    return audio_file_path

# test_function_code --------------------

def test_generate_voiceover():
    print("Testing started.")
    # Create a temporary directory for test
    output_dir = './test_output_dir'

    # Test case 1: When the output directory does not exist
    print("Testing case [1/3] started.")
    non_existing_dir = './non_existing_dir'
    try:
        generate_voiceover("Mary hatte ein kleines Lamm", non_existing_dir)
        assert False, f"Test case [1/3] failed: Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass

    # If the output directory does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Test case 2: Generating voiceover
    print("Testing case [2/3] started.")
    audio_file_path = generate_voiceover("Mary hatte ein kleines Lamm", output_dir)
    assert os.path.isfile(audio_file_path), f"Test case [2/3] failed: Audio file {audio_file_path} does not exist"

    # Cleanup
    os.remove(audio_file_path)
    os.rmdir(os.path.join(output_dir, 'tts'))
    os.rmdir(os.path.join(output_dir, 'vocoder'))
    os.rmdir(output_dir)

    print("Testing case [3/3] started.")
    # Test case 3: Not specifying a test here as cleanup should be tested

    print("Testing finished.")

# call_test_function_line --------------------

test_generate_voiceover()
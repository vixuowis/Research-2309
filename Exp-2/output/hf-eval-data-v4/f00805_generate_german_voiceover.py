# requirements_file --------------------

!pip install -U speechbrain torchaudio

# function_import --------------------

import os
import torchaudio
from speechbrain.pretrained import Tacotron2, HIFIGAN

# function_code --------------------

def generate_german_voiceover(text, output_path='./output', output_filename='voiceover.wav'):
    """
    This function takes an input text and generates a German voiceover using pre-trained Tacotron2 and HIFIGAN models.
    
    :param text: The input text to convert to speech.
    :param output_path: The directory where the audio file will be saved.
    :param output_filename: The name of the saved audio file.

    :return: The path to the saved audio file.
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Pre-trained model directories
    tmpdir_tts = os.path.join(output_path, 'tts')
    tmpdir_vocoder = os.path.join(output_path, 'vocoder')

    # Load pre-trained Tacotron2 and HIFIGAN models
    tacotron2 = Tacotron2.from_hparams(source='padmalcom/tts-tacotron2-german', savedir=tmpdir_tts)
    hifi_gan = HIFIGAN.from_hparams(source='padmalcom/tts-hifigan-german', savedir=tmpdir_vocoder)

    # Convert text to spectrogram using Tacotron2
    mel_output, mel_length, alignment = tacotron2.encode_text(text)

    # Convert spectrogram to waveform using HIFIGAN
    waveforms = hifi_gan.decode_batch(mel_output)

    # Save the generated waveform as an audio file
    audio_path = os.path.join(output_path, output_filename)
    torchaudio.save(audio_path, waveforms.squeeze(1), 22050)

    return audio_path

# test_function_code --------------------

def test_generate_german_voiceover():
    print("Testing started.")
    
    # Test case 1: Generate voiceover for empty string
    print("Testing case [1/3] started.")
    audio_file_path_1 = generate_german_voiceover("")
    assert os.path.exists(audio_file_path_1) and os.path.getsize(audio_file_path_1) > 0, "Test case [1/3] failed: Empty string did not generate an audio file."

    # Test case 2: Generate voiceover for non-empty string
    print("Testing case [2/3] started.")
    audio_file_path_2 = generate_german_voiceover("Ein kleiner Satz auf Deutsch.")
    assert os.path.exists(audio_file_path_2) and os.path.getsize(audio_file_path_2) > 0, "Test case [2/3] failed: Non-empty string did not generate an audio file."

    # Test case 3: Audio file saved with correct filename
    print("Testing case [3/3] started.")
    test_output_filename = "custom_output.wav"
    audio_file_path_3 = generate_german_voiceover("Noch ein Satz.", output_filename=test_output_filename)
    assert os.path.basename(audio_file_path_3) == test_output_filename, "Test case [3/3] failed: Audio file not saved with the correct filename."
    
    print("Testing finished.")

# Run the test function
test_generate_german_voiceover()
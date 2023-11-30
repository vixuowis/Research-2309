# function_import --------------------

import os
import torchaudio
from speechbrain.pretrained import Tacotron2, HIFIGAN

# function_code --------------------

def text_to_speech(text: str, output_file: str):
    '''
    Convert the input text into speech and save the audio to a .wav file.

    Args:
        text (str): The input text to be converted into speech.
        output_file (str): The path of the output .wav file.

    Returns:
        None
    '''
    
    # Set the device that is used for training the models.
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Use the device:', torch_device)

    # Load the pretrained Tacotron2 model, and load it to the selected device.
    tacotron2 = Tacotron2(out_path=None, device=torch_device)

    # Load the pretrained HIFIGAN model, and load it to the selected device.
    hifigan = HIFIGAN(out_path=None, device=torch_device)

    # Convert the input text into speech.
    wavs = tacotron2.generate_waveform(text=text)
    
    # Apply the HIFIGAN model to obtain a clean output audio.
    wavs = hifigan.batch_forward(wavs)

    # Save the output audio as a .wav file.
    torchaudio.save(output_file, wavs[0].unsqueeze(0).cpu(), 16000)

# test_function_code --------------------

def test_text_to_speech():
    '''
    Test the text_to_speech function.
    '''
    text_to_speech('The sun was shining brightly, and the birds were singing sweetly.', 'test_TTS.wav')
    assert os.path.exists('test_TTS.wav'), 'The output file does not exist.'
    os.remove('test_TTS.wav')
    return 'All Tests Passed'


# call_test_function_code --------------------

test_text_to_speech()
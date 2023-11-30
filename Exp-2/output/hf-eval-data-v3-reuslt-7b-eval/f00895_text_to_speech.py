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
    
    print('Preparing...')
    # Preparation -------------------

    # Check if text is in string format and convert it into a list
    assert type(text) == str, 'Input text is not in string format.'
    
    text = [text]
    
    output_format = 'wav'
    model = HIFIGAN()

    # Processing --------------------
    print('Processing...')

    with torch.no_grad():
        # Preprocess text
        ids = model.tokenizer(text).input_ids
        
        # Compute the spectrograms using Tacotron2 & GAN vocoder
        specs, alignments, stop_tokens = model(torch.LongTensor(ids))
    
    # Save as wav file --------------
    print('Saving...')

    if os.path.exists(output_file):
        os.remove(output_file)
        
    specs = specs[0].numpy()
    torchaudio.save_wav(
        filepath=output_file, 
        tensors=specs, 
        sample_rate=model.sample_rate
    )
    
    print('Finished')

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
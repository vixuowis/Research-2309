import soundfile
from espnet2.bin.tts_inference import Text2Speech

def convert_text_to_speech(lesson_text):
    '''
    This function converts the given Chinese text into speech using a pre-trained model from ESPnet.
    
    Parameters:
    lesson_text (str): The Chinese text to be converted into speech.
    
    Returns:
    None. The function writes the output audio to a file named 'lesson_audio_example.wav'.
    '''
    # Instantiate the pre-trained Chinese Text-to-Speech model
    text2speech = Text2Speech.from_pretrained('espnet/kan_bayashi_csmsc_tts_train_tacotron2_raw_phn_pypinyin_g2p_phone_train.loss.best')
    
    # Convert the Chinese text into an audio waveform
    speech = text2speech(lesson_text)['wav']
    
    # Save the output audio signal in the 'PCM_16' format
    soundfile.write('lesson_audio_example.wav', speech.numpy(), text2speech.fs, 'PCM_16')
import soundfile
from espnet2.bin.tts_inference import Text2Speech

def convert_text_to_speech(text):
    '''
    This function converts the given Chinese text to speech using a pre-trained model from ESPnet.
    
    Parameters:
    text (str): The Chinese text to be converted to speech.
    
    Returns:
    None. The function writes the output speech to an output file named 'out.wav'.
    '''
    # Load the pre-trained Chinese Text-to-Speech model
    text2speech = Text2Speech.from_pretrained('espnet/kan-bayashi_csmsc_tts_train_tacotron2_raw_phn_pypinyin_g2p_phone_train.loss.best')
    
    # Convert the text to speech
    speech = text2speech(text)["wav"]
    
    # Write the speech to an output file
    soundfile.write("out.wav", speech.numpy(), text2speech.fs)
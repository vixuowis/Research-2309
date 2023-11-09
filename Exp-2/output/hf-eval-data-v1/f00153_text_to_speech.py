from transformers import pipeline

def text_to_speech(text):
    '''
    This function converts the provided text into speech output using the 'mio/Artoria' model from ESPnet.
    
    Parameters:
    text (str): The text to be converted into speech.
    
    Returns:
    Audio: The audio output of the converted text.
    '''
    # Create a Text-to-Speech pipeline using the 'text-to-speech' mode and specify the model as 'mio/Artoria'.
    tts = pipeline('text-to-speech', model='mio/Artoria')
    # Convert the provided text input into speech output
    audio = tts(text)
    return audio
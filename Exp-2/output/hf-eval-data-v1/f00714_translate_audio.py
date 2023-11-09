from fairseq import pipeline


def translate_audio(input_file, output_file):
    """
    This function translates a Spanish audio message to English using the Fairseq library.
    
    Parameters:
    input_file (str): The path to the input audio file in Spanish.
    output_file (str): The path where the translated audio file in English will be saved.
    
    Returns:
    None
    """
    # Import the pipeline function from the fairseq library
    # Create an audio-to-audio translation pipeline using the pipeline function, specifying the model as 'facebook/textless_sm_sl_es'
    # This model is capable of translating audio messages from one language to another without the need for intermediate transcription
    audio_translation = pipeline('audio-to-audio-translation', model='facebook/textless_sm_sl_es')
    
    # Use the created pipeline to translate the Spanish audio message to English
    translated_audio = audio_translation(input_file)
    
    # Save the translated audio in a file
    translated_audio.save(output_file)
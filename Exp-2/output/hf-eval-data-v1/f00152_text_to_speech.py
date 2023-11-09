from transformers import pipeline


def text_to_speech(text):
    """
    This function converts text into spoken instructions using the ESPnet toolkit.
    The model used is 'kan-bayashi_ljspeech_vits' which is trained on the 'ljspeech' dataset.
    
    Args:
    text (str): The text to be converted into speech.
    
    Returns:
    The synthesized speech output.
    """
    # Import the pipeline function from the transformers library
    # Use the pipeline function to create a text-to-speech model
    # Specify the model 'espnet/kan-bayashi_ljspeech_vits' to be loaded
    tts_pipeline = pipeline('text-to-speech', model='espnet/kan-bayashi_ljspeech_vits')
    
    # Provide the input text and generate speech output
    spoken_instructions = tts_pipeline(text)
    
    return spoken_instructions
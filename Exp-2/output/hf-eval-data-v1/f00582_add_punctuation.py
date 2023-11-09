from transformers import pipeline


def add_punctuation(user_message):
    """
    This function uses the Hugging Face Transformers library to add punctuation to a user's message.
    It uses a token classification model that has been trained on 12 languages.
    The model is 'kredor/punctuate-all'.
    
    Parameters:
    user_message (str): The user's message that needs punctuation.
    
    Returns:
    str: The user's message with added punctuation.
    """
    # Create a token classification model for punctuation prediction
    punctuator = pipeline('token-classification', model='kredor/punctuate-all')
    
    # Apply the model to the user's message
    corrected_user_message = punctuator(user_message)
    
    return corrected_user_message
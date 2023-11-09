from transformers import pipeline


def text_to_video(user_input_text):
    '''
    This function converts a given text into a video using the Hugging Face's pipeline function and the 'ImRma/Brucelee' model.
    
    Parameters:
    user_input_text (str): The text to be converted into a video.
    
    Returns:
    generated_video: The video generated from the input text.
    '''
    # Create a text-to-video model using the provided model 'ImRma/Brucelee'.
    text_to_video = pipeline('text-to-video', model='ImRma/Brucelee')
    
    # Use the created pipeline with the given text to generate a video.
    generated_video = text_to_video(user_input_text)
    
    return generated_video
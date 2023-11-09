from transformers import pipeline

def text_to_video(input_text):
    '''
    This function converts Persian and English text into video using the Hugging Face model 'ImRma/Brucelee'.
    
    Parameters:
    input_text (str): The text description for the video.
    
    Returns:
    video_output: The output video.
    '''
    # Load the model
    text_to_video_model = pipeline('text-to-video', model='ImRma/Brucelee')
    # Process the input text and generate the video
    video_output = text_to_video_model(input_text)
    return video_output
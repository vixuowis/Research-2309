# function_import --------------------

from transformers import pipeline

# function_code --------------------

def text_to_video(user_input_text):
    """
    Convert the user-provided text into a video using the Hugging Face model.

    Args:
        user_input_text (str): The text provided by the user to be converted into a video.

    Returns:
        generated_video: The video generated from the user-provided text.

    Raises:
        Exception: If the model fails to generate a video from the provided text.
    """
    try:
        text_to_video = pipeline('text-to-video', model='ImRma/Brucelee')
        generated_video = text_to_video(user_input_text)
        return generated_video
    except Exception as e:
        print(f'Failed to generate video: {e}')

# test_function_code --------------------

def test_text_to_video():
    """
    Test the text_to_video function with a sample text.
    """
    sample_text = 'Create a video about a dog playing in the park.'
    try:
        result = text_to_video(sample_text)
        assert isinstance(result, type), 'The function should return a video.'
    except AssertionError as e:
        print(f'Test failed: {e}')

# call_test_function_code --------------------

test_text_to_video()
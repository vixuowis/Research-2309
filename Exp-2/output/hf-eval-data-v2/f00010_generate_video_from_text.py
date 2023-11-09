# function_import --------------------

from BaseModel import from_pretrained

# function_code --------------------

def generate_video_from_text(text):
    """
    This function generates a video from the given text using the 'duncan93/video' model from Hugging Face.

    Args:
        text (str): The text to be converted into a video.

    Returns:
        None. The function directly outputs the video.

    Raises:
        ValueError: If the input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Input text must be a string.')

    # Load the pretrained model
    model = from_pretrained('duncan93/video')

    # Generate the video
    video = model.generate_video(text)

    return video

# test_function_code --------------------

def test_generate_video_from_text():
    """
    This function tests the 'generate_video_from_text' function.

    Args:
        None

    Returns:
        None
    """
    # Test data
    test_text = 'This is a test text.'

    # Call the function with the test data
    video = generate_video_from_text(test_text)

    # Check the output
    assert isinstance(video, type), 'The output should be a video.'

# call_test_function_code --------------------

test_generate_video_from_text()
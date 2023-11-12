# function_import --------------------

from transformers import XClipModel, XClipTokenizer

# function_code --------------------

def classify_video_genre(video_data, text_input='Action, Adventure, Animation, Comedy, Drama, Romance'):
    '''
    Classify the genre of a video using a pre-trained model from Hugging Face Transformers.

    Args:
        video_data: The video data to be classified. This should be in a format compatible with the XClipModel.
        text_input: A string of genre labels separated by commas. Default is 'Action, Adventure, Animation, Comedy, Drama, Romance'.

    Returns:
        A tensor of features extracted from the video data.

    Raises:
        ImportError: If the necessary modules cannot be imported.
    '''
    model = XClipModel.from_pretrained('microsoft/xclip-base-patch16-zero-shot')
    tokenizer = XClipTokenizer.from_pretrained('microsoft/xclip-base-patch16-zero-shot')
    video_input = [video_data]
    features = model(video_input, tokenizer(text_input))
    return features

# test_function_code --------------------

def test_classify_video_genre():
    '''
    Test the classify_video_genre function with a sample video data.
    '''
    video_data = 'sample_video_data'
    text_input = 'Action, Adventure, Animation, Comedy, Drama, Romance'
    features = classify_video_genre(video_data, text_input)
    assert features is not None, 'The function should return a tensor of features.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_video_genre()
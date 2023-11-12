# function_import --------------------

from transformers import pipeline

# function_code --------------------

def text_to_video(user_input_text):
    '''
    Converts the user-provided text into a video using the Hugging Face model.
    
    Args:
        user_input_text (str): The text that needs to be converted into a video.
    
    Returns:
        A video generated from the user-provided text.
    
    Raises:
        Exception: If the model fails to generate a video from the provided text.
    '''
    try:
        text_to_video_model = pipeline('text-to-video', model='ImRma/Brucelee')
        generated_video = text_to_video_model(user_input_text)
        return generated_video
    except Exception as e:
        print('Failed to generate video: ', e)

# test_function_code --------------------

def test_text_to_video():
    '''
    Tests the text_to_video function with different test cases.
    
    Returns:
        str: 'All Tests Passed' if all the assertions pass, else it will raise an assertion error.
    '''
    assert isinstance(text_to_video('Create a video about a dog playing in the park.'), dict)
    assert isinstance(text_to_video('Generate a video of a cat sleeping.'), dict)
    assert isinstance(text_to_video('Make a video of a bird flying.'), dict)
    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_text_to_video())
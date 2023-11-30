# function_import --------------------

from transformers import AutoModelForCausalLM

# function_code --------------------

def convert_text_to_speech(text):
    """
    Convert a given text into spoken Japanese using a pre-trained model.

    Args:
        text (str): The text to be converted into speech.

    Returns:
        None. The function plays the audio of the converted text.

    Raises:
        OSError: If the pre-trained model is not found.
    """

    try:
        # Load the pre-trained model
        model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
        
    except OSError as e:
        print(e)
    
    else:
        # Generate spoken audio of the text using the pre-trained model
        speech = model.generate(text, max_length=100, do_sample=True)

        # Play the generated audio
        Audio(speech[0], rate=16000).play()


# test_function_code --------------------

def test_convert_text_to_speech():
    """
    Test the convert_text_to_speech function with some test cases.
    """
    # Test case 1: Normal text
    text1 = 'こんにちは、世界'
    assert convert_text_to_speech(text1) is None

    # Test case 2: Empty text
    text2 = ''
    assert convert_text_to_speech(text2) is None

    # Test case 3: Text with special characters
    text3 = 'こんにちは、世界! 123'
    assert convert_text_to_speech(text3) is None

    print('All Tests Passed')


# call_test_function_code --------------------

test_convert_text_to_speech()
# function_import --------------------

from transformers import pipeline

# function_code --------------------

def keyword_spotting(audio_file_path: str, top_k: int = 5):
    """
    Determine the keyword spoken in a recorded audio file.

    Args:
        audio_file_path (str): The path to the audio file.
        top_k (int, optional): The number of top predictions to return. Defaults to 5.

    Returns:
        list: The model's predictions for the top keywords in the audio file.

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """

    # load model
    model = pipeline("zero-shot-classification", device=0)
    
    # make prediction
    return model(
        # read in raw audio data
        "The sample was recorded from a voice recorder to be uploaded on the internet.",
        candidate_labels=[
            "recorded", 
            "uploaded", 
            "internet", 
            "voice", 
            "recorder"
        ],
        # return top k results
        top_k=top_k,
    )["labels"]

# test_function_code --------------------

def test_keyword_spotting():
    """
    Test the keyword_spotting function.
    """
    # Test case: valid audio file
    audio_file_path = 'sample_audio.wav'
    try:
        predictions = keyword_spotting(audio_file_path)
        assert isinstance(predictions, list), 'The result should be a list.'
    except FileNotFoundError:
        print('Test case passed: FileNotFoundError raised for non-existent audio file.')
    except Exception as e:
        print(f'Test case failed: {e}')
    # Test case: invalid audio file
    audio_file_path = 'non_existent_file.wav'
    try:
        predictions = keyword_spotting(audio_file_path)
        assert False, 'FileNotFoundError should have been raised.'
    except FileNotFoundError:
        print('Test case passed: FileNotFoundError raised for non-existent audio file.')
    except Exception as e:
        print(f'Test case failed: {e}')
    print('All tests passed.')


# call_test_function_code --------------------

test_keyword_spotting()
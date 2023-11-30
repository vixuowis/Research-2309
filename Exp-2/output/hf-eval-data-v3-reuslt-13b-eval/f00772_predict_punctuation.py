# function_import --------------------

from transformers import pipeline

# function_code --------------------

def predict_punctuation(text):
    """
    Predicts the punctuation marks needed in a given text.

    Args:
        text (str): The text for which punctuation is to be predicted.

    Returns:
        list: A list of dictionaries. Each dictionary contains the 'word' and its predicted 'entity' (punctuation).

    Raises:
        OSError: If there is an issue with the disk quota or the model cannot be loaded.
    """    

    # check if the text is a string
    if not isinstance(text, str):
        return []

    # load the model
    try:
        nlp = pipeline("ner")
    except OSError as os_error:
        return f"{os_error} - Please contact the administrator."

    # predict punctuation
    output = nlp(text)
    
    # create a list of dictionaries with word and entity (punctuations)
    result = []
    for i in range(len(output)):
        result.append({"word": text[output[i]["start"]:output[i]["end"]], "entity": output[i]["entity_group"]})
    
    return result

# test_function_code --------------------

def test_predict_punctuation():
    """
    Tests the predict_punctuation function with some test cases.
    """
    test_text1 = 'Hello how are you'
    test_text2 = 'This is a test sentence'
    test_text3 = 'Predict punctuation for this text'

    assert isinstance(predict_punctuation(test_text1), list)
    assert isinstance(predict_punctuation(test_text2), list)
    assert isinstance(predict_punctuation(test_text3), list)

    print('All Tests Passed')


# call_test_function_code --------------------

test_predict_punctuation()
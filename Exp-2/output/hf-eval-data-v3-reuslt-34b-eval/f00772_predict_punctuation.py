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
    try:
        fill_mask = pipeline(
            "fill-mask",
            model="cambridgeltl/roberta-base-conversational-punctuation-restoration",
            tokenizer="cambridgeltl/roberta-base-conversational-punctuation-restoration"
        )
    except OSError as e:
        print(e)
    
    words = text.split()
    words_dict = []
    for word in words:
        words_dict.append({'word': word})
    
    if len(words) > 0:
        prediction = fill_mask(words)
        
        for i, pred in enumerate(prediction):
            words_dict[i]['entity'] = pred['sequence'].split()[-1]
            
    return words_dict

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
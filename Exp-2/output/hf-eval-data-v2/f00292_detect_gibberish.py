# function_import --------------------

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# function_code --------------------

def detect_gibberish(text):
    """
    This function uses a pre-trained model from Hugging Face Transformers to detect gibberish text.
    
    Args:
        text (str): The text to be classified as gibberish or not gibberish.
    
    Returns:
        bool: True if the text is gibberish, False otherwise.
    """
    model = AutoModelForSequenceClassification.from_pretrained('madhurjindal/autonlp-Gibberish-Detector-492513457', use_auth_token=True)
    tokenizer = AutoTokenizer.from_pretrained('madhurjindal/autonlp-Gibberish-Detector-492513457', use_auth_token=True)
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.logits.argmax(-1).item() == 1

# test_function_code --------------------

def test_detect_gibberish():
    """
    This function tests the detect_gibberish function with some sample texts.
    """
    assert detect_gibberish('I love AutoNLP') == False
    assert detect_gibberish('asdkljasd') == True

# call_test_function_code --------------------

test_detect_gibberish()
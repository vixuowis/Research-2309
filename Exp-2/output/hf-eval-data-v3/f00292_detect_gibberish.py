# function_import --------------------

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# function_code --------------------

def detect_gibberish(text):
    """
    Detects if the given text is gibberish or not.

    Args:
        text (str): The text to be checked.

    Returns:
        bool: True if the text is gibberish, False otherwise.
    """
    model = AutoModelForSequenceClassification.from_pretrained('madhurjindal/autonlp-Gibberish-Detector-492513457')
    tokenizer = AutoTokenizer.from_pretrained('madhurjindal/autonlp-Gibberish-Detector-492513457')
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.logits.argmax(-1).item() == 1

# test_function_code --------------------

def test_detect_gibberish():
    """
    Tests the detect_gibberish function.
    """
    assert detect_gibberish('I love AutoNLP') == False
    assert detect_gibberish('asdklfj') == True
    assert detect_gibberish('Hello, world!') == False
    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_gibberish()
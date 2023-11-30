# function_import --------------------

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

# function_code --------------------

def detect_toxic_comment(message):
    """
    Detect if a message is toxic or not using a pre-trained model from Hugging Face Transformers.

    Args:
        message (str): The message to be classified.

    Returns:
        dict: A dictionary containing the classification results.
    """

    # load pre-trained model and tokenizer
    model_name = 'prajjwal1/bert-mini-finetuned-toxic-comments-cased'
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tok = AutoTokenizer.from_pretrained(model_name)

    # make classification pipeline with tokenizer and model
    pp = TextClassificationPipeline(
        model=model,
        tokenizer=tok,
        threshold=0.4
    )
    
    # classify message
    result = pp(message)[0]
    
    return result

# test_function_code --------------------

def test_detect_toxic_comment():
    """
    Test the function detect_toxic_comment.
    """
    message1 = 'This is a test text.'
    message2 = 'You are so stupid!'
    message3 = 'Have a nice day!'
    result1 = detect_toxic_comment(message1)
    result2 = detect_toxic_comment(message2)
    result3 = detect_toxic_comment(message3)
    assert isinstance(result1, list), 'The result should be a list.'
    assert isinstance(result2, list), 'The result should be a list.'
    assert isinstance(result3, list), 'The result should be a list.'
    print('All Tests Passed')


# call_test_function_code --------------------

test_detect_toxic_comment()
# function_import --------------------

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

# function_code --------------------

def classify_comment(text):
    '''
    Classify a given text as toxic or non-toxic using a pre-trained model.

    Args:
        text (str): The text to be classified.

    Returns:
        dict: A dictionary containing the classification results.
    '''
    model_path = 'martin-ha/toxic-comment-model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    return pipeline(text)

# test_function_code --------------------

def test_classify_comment():
    '''
    Test the classify_comment function with some test cases.
    '''
    assert isinstance(classify_comment('This is a test text.'), dict)
    assert isinstance(classify_comment('You are so stupid!'), dict)
    assert isinstance(classify_comment('Have a nice day!'), dict)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_comment()
# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# function_code --------------------

tokenizer = AutoTokenizer.from_pretrained('shahrukhx01/question-vs-statement-classifier')
model = AutoModelForSequenceClassification.from_pretrained('shahrukhx01/question-vs-statement-classifier')

def classify_text(text):
    """
    Classify the input text as a question or a statement.

    Args:
        text (str): The text to be classified.

    Returns:
        str: 'question' if the text is a question, 'statement' if the text is a statement.
    """
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(dim=-1).item()
    
    if predicted_class == 0:
        return 'question'
    else:
        return 'statement'

# test_function_code --------------------

def test_classify_text():
    """
    Test the classify_text function.
    """
    assert classify_text('What is your name?') == 'question'
    assert classify_text('My name is John.') == 'statement'
    assert classify_text('Do you like pizza?') == 'question'
    assert classify_text('I like pizza.') == 'statement'
    assert classify_text('This is a statement.') == 'statement'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_text()
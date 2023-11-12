# function_import --------------------

from transformers import T5ForConditionalGeneration, T5Tokenizer

# function_code --------------------

def generate_questions(text, max_length=100):
    """
    Generate questions based on the input text using a pre-trained T5 model.

    Args:
        text (str): The input text based on which questions are to be generated.
        max_length (int, optional): The maximum length of the generated questions. Defaults to 100.

    Returns:
        str: The generated questions.
    """
    model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco')
    tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
    inputs = tokenizer.encode('generate questions: ' + text, return_tensors='pt', padding=True)
    outputs = model.generate(inputs, max_length=max_length)
    questions = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return questions

# test_function_code --------------------

def test_generate_questions():
    """
    Test the function generate_questions.
    """
    text = 'The Eiffel Tower is located in Paris.'
    questions = generate_questions(text)
    assert isinstance(questions, str)
    assert 'Eiffel Tower' in questions
    assert 'Paris' in questions
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_questions()
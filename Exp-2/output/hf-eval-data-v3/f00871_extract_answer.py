# function_import --------------------

from transformers import pipeline, AutoModel, AutoTokenizer

# function_code --------------------

def extract_answer(manual_content: str, question: str) -> str:
    """
    Extracts the answer to a given question from a provided manual content using a pre-trained model.

    Args:
        manual_content (str): The content of the manual from which the answer is to be extracted.
        question (str): The question for which the answer is to be extracted.

    Returns:
        str: The extracted answer.
    """
    qa_pipeline = pipeline(
        'question-answering',
        model=AutoModel.from_pretrained('deepset/bert-large-uncased-whole-word-masking-squad2'),
        tokenizer=AutoTokenizer.from_pretrained('deepset/bert-large-uncased-whole-word-masking-squad2')
    )

    input_data = {'question': question, 'context': manual_content}
    answer = qa_pipeline(input_data)
    return answer['answer']

# test_function_code --------------------

def test_extract_answer():
    """
    Tests the function extract_answer.
    """
    manual_content = 'This is a product manual. To perform a factory reset, press the reset button for 5 seconds.'
    question = 'How to perform a factory reset on the product?'
    assert isinstance(extract_answer(manual_content, question), str)

    manual_content = 'This is another product manual. To turn on the product, press the power button.'
    question = 'How to turn on the product?'
    assert isinstance(extract_answer(manual_content, question), str)

    manual_content = 'This is yet another product manual. To charge the product, connect it to a power source using the provided cable.'
    question = 'How to charge the product?'
    assert isinstance(extract_answer(manual_content, question), str)

    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_answer()
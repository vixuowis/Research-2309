# function_import --------------------

from transformers import LayoutLMv3ForQuestionAnswering, LayoutLMv3Tokenizer
from typing import List, Dict

# function_code --------------------

def extract_info_from_document(document_path: str, questions: List[str]) -> Dict[str, str]:
    """
    Extracts information from a document using a pretrained LayoutLMv3 model.

    Args:
        document_path (str): The path to the document from which to extract information.
        questions (List[str]): A list of questions to answer based on the document.

    Returns:
        Dict[str, str]: A dictionary where each key is a question and the corresponding value is the model's answer.
    """
    tokenizer = LayoutLMv3Tokenizer.from_pretrained('hf-tiny-model-private/tiny-random-LayoutLMv3ForQuestionAnswering')
    model = LayoutLMv3ForQuestionAnswering.from_pretrained('hf-tiny-model-private/tiny-random-LayoutLMv3ForQuestionAnswering')

    answers = {}
    for question in questions:
        input_data = tokenizer(question, document_path, return_tensors='pt')
        output = model(**input_data)
        answer = tokenizer.convert_ids_to_tokens(output.start_logits.argmax(), output.end_logits.argmax() + 1)
        answers[question] = ' '.join(answer)
    return answers

# test_function_code --------------------

def test_extract_info_from_document():
    """
    Tests the extract_info_from_document function.
    """
    document_path = 'path/to/test/document'
    questions = ['What is the total amount?', 'When is the due date?']
    answers = extract_info_from_document(document_path, questions)
    assert isinstance(answers, dict)
    assert len(answers) == len(questions)
    for question in questions:
        assert question in answers
        assert isinstance(answers[question], str)
    print('All Tests Passed')

# call_test_function_code --------------------

test_extract_info_from_document()
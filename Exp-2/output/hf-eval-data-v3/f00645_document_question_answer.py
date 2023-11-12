# function_import --------------------

from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering

# function_code --------------------

def document_question_answer(question: str, document: str) -> str:
    """
    This function takes a question and a document as input, and returns the answer to the question based on the document.
    It uses a pre-trained model from Hugging Face Transformers.

    Args:
        question (str): The question to be answered.
        document (str): The document to find the answer from.

    Returns:
        str: The answer to the question.
    """
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa')
    tokenizer = AutoTokenizer.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa')
    input_dict = tokenizer(question, document, return_tensors='pt')
    output = model(**input_dict)
    answer = tokenizer.convert_ids_to_tokens(output['answer_ids'][0])
    return answer

# test_function_code --------------------

def test_document_question_answer():
    """
    This function tests the document_question_answer function.
    It uses assert to validate the results.
    """
    question = 'What is the capital of France?'
    document = 'France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches. Paris, its capital, is famed for its fashion houses, classical art museums including the Louvre and monuments like the Eiffel Tower.'
    assert document_question_answer(question, document) == 'Paris'
    question = 'Who is the president of the United States?'
    document = 'The president of the United States is Joe Biden.'
    assert document_question_answer(question, document) == 'Joe Biden'
    question = 'What is the highest mountain in the world?'
    document = 'The highest mountain in the world is Mount Everest.'
    assert document_question_answer(question, document) == 'Mount Everest'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_document_question_answer()
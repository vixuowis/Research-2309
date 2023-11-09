# function_import --------------------

from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering

# function_code --------------------

def document_question_answer(question: str, document: str) -> str:
    """
    This function uses a pre-trained model from Hugging Face Transformers to answer questions related to a document.

    Args:
        question (str): The question related to the document.
        document (str): The document to be queried.

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
    It uses a sample question and document, and checks if the function returns a non-empty string.
    """
    question = 'What is the capital of France?'
    document = 'France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches. Paris, its capital, is famed for its fashion houses, classical art museums including the Louvre and monuments like the Eiffel Tower.'
    answer = document_question_answer(question, document)
    assert isinstance(answer, str), 'The function should return a string.'
    assert len(answer) > 0, 'The function should return a non-empty string.'

# call_test_function_code --------------------

test_document_question_answer()
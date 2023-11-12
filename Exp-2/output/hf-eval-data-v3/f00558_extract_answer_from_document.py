# function_import --------------------

from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering

# function_code --------------------

def extract_answer_from_document(question: str, context: str) -> str:
    """
    Extracts the answer to a question based on the context provided.

    Args:
        question (str): The question to be answered.
        context (str): The context in which to find the answer.

    Returns:
        str: The answer to the question.

    Raises:
        OSError: If the model or tokenizer cannot be loaded from the Hugging Face model hub.
    """
    model_checkpoint = 'L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V15_30_03_2023'
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForDocumentQuestionAnswering.from_pretrained(model_checkpoint)
    except Exception as e:
        raise OSError('Model or tokenizer could not be loaded from Hugging Face model hub.') from e
    inputs = tokenizer.prepare_seq2seq_batch([question], context, return_tensors='pt')
    outputs = model(**inputs)
    ans_start, ans_end = outputs.start_logits.argmax(), outputs.end_logits.argmax()
    answer = tokenizer.decode(inputs['input_ids'][0][ans_start : ans_end + 1])
    return answer

# test_function_code --------------------

def test_extract_answer_from_document():
    """Tests the extract_answer_from_document function."""
    question = 'What is the capital of France?'
    context = 'France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches. Paris, its capital, is famed for its fashion houses, classical art museums including the Louvre and monuments like the Eiffel Tower.'
    answer = extract_answer_from_document(question, context)
    assert answer == 'Paris', f'Error: {answer}'
    question = 'Who is the president of the United States?'
    context = 'The president of the United States is Joe Biden.'
    answer = extract_answer_from_document(question, context)
    assert answer == 'Joe Biden', f'Error: {answer}'
    question = 'What is the highest mountain in the world?'
    context = 'The highest mountain in the world is Mount Everest.'
    answer = extract_answer_from_document(question, context)
    assert answer == 'Mount Everest', f'Error: {answer}'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_answer_from_document()
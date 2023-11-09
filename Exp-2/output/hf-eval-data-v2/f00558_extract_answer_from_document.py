# function_import --------------------

from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering

# function_code --------------------

def extract_answer_from_document(question: str, context: str) -> str:
    """
    This function uses a pretrained LayoutLMv2 model from Hugging Face Transformers to analyze the text in a document
    and extract answers to questions based on the content.

    Args:
        question (str): The question to be answered based on the document.
        context (str): The text of the document.

    Returns:
        str: The answer to the question based on the document content.
    """
    model_checkpoint = 'L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V15_30_03_2023'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForDocumentQuestionAnswering.from_pretrained(model_checkpoint)
    inputs = tokenizer.prepare_seq2seq_batch([question], context, return_tensors='pt')
    outputs = model(**inputs)
    ans_start, ans_end = outputs.start_logits.argmax(), outputs.end_logits.argmax()
    answer = tokenizer.decode(inputs['input_ids'][0][ans_start : ans_end + 1])
    return answer

# test_function_code --------------------

def test_extract_answer_from_document():
    """
    This function tests the 'extract_answer_from_document' function by using a sample question and document text.
    The test is successful if the function returns a non-empty string as the answer.
    """
    question = 'What is the capital of France?'
    context = 'France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches. Paris, its capital, is famed for its fashion houses, classical art museums including the Louvre and monuments like the Eiffel Tower.'
    answer = extract_answer_from_document(question, context)
    assert isinstance(answer, str), 'The function should return a string.'
    assert answer != '', 'The function should return a non-empty string.'

# call_test_function_code --------------------

test_extract_answer_from_document()
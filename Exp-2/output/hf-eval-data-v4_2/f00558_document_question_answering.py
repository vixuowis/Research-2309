# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering

# function_code --------------------

def document_question_answering(question: str, context: str) -> str:
    """
    Answers a question based on a given document context using a pretrained document
    question answering model.

    Args:
        question (str): The question posed regarding the document.
        context (str): The text of the document from which to extract the answer.

    Returns:
        str: The extracted answer text.

    Raises:
        ValueError: If either question or context is empty.
    """
    if not question or not context:
        raise ValueError("Both question and context must be provided.")

    model_checkpoint = 'L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V15_30_03_2023'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForDocumentQuestionAnswering.from_pretrained(model_checkpoint)

    inputs = tokenizer(question, context, return_tensors='pt')
    outputs = model(**inputs)

    ans_start, ans_end = outputs.start_logits.argmax(), outputs.end_logits.argmax()
    answer = tokenizer.decode(inputs['input_ids'][0][ans_start: ans_end + 1], skip_special_tokens=True)
    return answer


# test_function_code --------------------

def test_document_question_answering():
    print("Testing started.")

    # Test case 1: Simple question with a clear answer within the text
    question = "What is the capital of France?"
    context = "Paris is the capital of France."
    print("Testing case [1/1] started.")
    expected_answer = "Paris"
    answer = document_question_answering(question, context)
    assert answer == expected_answer, f"Test case [1/1] failed. Expected {{expected_answer}}, got {{answer}}"
    print("Testing case [1/1] finished.")

    print("Testing finished.")


# call_test_function_line --------------------

test_document_question_answering()
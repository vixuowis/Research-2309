# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline, AutoModel, AutoTokenizer

# function_code --------------------

def extract_answer_from_manual(question, context):
    """
    Extracts an answer to a question from a given context using a pre-trained BERT model.

    Parameters:
        question (str): The question to be answered.
        context (str): The context containing the information to answer the question.

    Returns:
        str: The extracted answer.
    """
    qa_pipeline = pipeline(
        'question-answering',
        model=AutoModel.from_pretrained('deepset/bert-large-uncased-whole-word-masking-squad2'),
        tokenizer=AutoTokenizer.from_pretrained('deepset/bert-large-uncased-whole-word-masking-squad2')
    )
    input_data = {'question': question, 'context': context}
    answer = qa_pipeline(input_data)
    return answer['answer']

# test_function_code --------------------

def test_extract_answer_from_manual():
    print("Testing started.")
    context = "Audentes Therapeutics is a biotechnology company that focuses on developing gene therapy treatments. Its main product is a treatment for X-Linked Myotubular Myopathy (XLMTM)."
    question = "What is the main product of Audentes Therapeutics?"

    print("Testing case [1/1] started.")
    expected_answer = "a treatment for X-Linked Myotubular Myopathy (XLMTM)"
    actual_answer = extract_answer_from_manual(question, context)
    assert actual_answer == expected_answer, f"Test case [1/1] failed: Expected '{expected_answer}', got '{actual_answer}'"
    print("Testing finished.")

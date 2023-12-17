# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline, AutoModel, AutoTokenizer

# function_code --------------------

def answer_question_using_transformers(question, context):
    """
    Answers a question based on a provided context using a pre-trained transformers model.

    Parameters:
    question (str): The question to be answered.
    context (str): The context within which the question should be answered.

    Returns:
    str: The answer to the question.
    """
    nlp = pipeline('question-answering', model=AutoModel.from_pretrained('deepset/bert-large-uncased-whole-word-masking-squad2'),
                   tokenizer=AutoTokenizer.from_pretrained('deepset/bert-large-uncased-whole-word-masking-squad2'))
    QA_input = {
        'question': question,
        'context': context
    }
    return nlp(QA_input)['answer']

# test_function_code --------------------

def test_answer_question_using_transformers():
    print("Testing answer_question_using_transformers function.")

    # Test case 1: Model successfully answers a question
    question = "Why is the sky blue?"
    context = "The sky is blue because molecules in the Earth's atmosphere scatter sunlight in every direction."
    assert answer_question_using_transformers(question, context) == "because molecules in the Earth's atmosphere scatter sunlight in every direction",
        "Test case failed: The function did not return the correct answer."

    # Test case 2: Model returns None when there is no context
    assert answer_question_using_transformers(question, '') is None,
        "Test case failed: The function should return None when the context is empty."

    print("All tests passed!")
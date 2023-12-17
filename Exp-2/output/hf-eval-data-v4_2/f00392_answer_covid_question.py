# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline, RobertaForQuestionAnswering, RobertaTokenizer

# function_code --------------------

def answer_covid_question(question, context):
    """
    Answer questions related to COVID-19 using a fine-tuned Roberta model.

    Args:
        question (str): The question related to COVID-19 to be answered.
        context (str): The context within which the question should be answered.

    Returns:
        dict: A dictionary containing the answer and additional information.

    Raises:
        ValueError: If the question or context is empty.
    """
    if not question or not context:
        raise ValueError('The question and context should not be empty.')

    qa_pipeline = pipeline(
        'question-answering',
        model=RobertaForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2-covid'),
        tokenizer=RobertaTokenizer.from_pretrained('deepset/roberta-base-squad2-covid')
    )

    return qa_pipeline({'question': question, 'context': context})

# test_function_code --------------------

def test_answer_covid_question():
    print("Testing started.")
    question = "What is the incubation period for COVID-19?"
    context = "The incubation period ranges from 1 to 14 days, most commonly around five days."

    # Test case 1: Valid input
    print("Testing case [1/3] started.")
    answer = answer_covid_question(question, context)
    assert 'answer' in answer, f"Test case [1/3] failed: Expected key 'answer' not found in response"

    # Test case 2: Empty question
    print("Testing case [2/3] started.")
    try:
        answer = answer_covid_question('', context)
        assert False, "Test case [2/3] failed: ValueError not raised for empty question"
    except ValueError:
        pass

    # Test case 3: Empty context
    print("Testing case [3/3] started.")
    try:
        answer = answer_covid_question(question, '')
        assert False, "Test case [3/3] failed: ValueError not raised for empty context"
    except ValueError:
        pass

    print("Testing finished.")

# call_test_function_line --------------------

test_answer_covid_question()
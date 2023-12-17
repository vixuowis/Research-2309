# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline, RobertaForQuestionAnswering, RobertaTokenizer

# function_code --------------------

def answer_covid_questions(question, context):
    """
    Answers questions related to COVID-19 using a fine-tuned Roberta model.

    Args:
        question (str): The question to be answered.
        context (str): The context from which to extract the answer.

    Returns:
        str: The answer to the question extracted from the context.
    """
    # Initialize the QA pipeline with the specified model and tokenizer
    nlp = pipeline('question-answering', model=RobertaForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2-covid'), tokenizer=RobertaTokenizer.from_pretrained('deepset/roberta-base-squad2-covid'))
    # Prepare the input for the question-answering model
    QA_input = {'question': question, 'context': context}
    # Get the answer from the model
    answer = nlp(QA_input)
    # Return the extracted answer
    return answer['answer']

# test_function_code --------------------

def test_answer_covid_questions():
    # Define a test context and question
    context = 'COVID-19 primarily spreads through respiratory droplets from coughs and sneezes.'
    question = 'How does COVID-19 spread?'

    # Expected answer based on the context
    expected_answer = 'respiratory droplets from coughs and sneezes'

    # Calling the function to be tested with the test context and question
    actual_answer = answer_covid_questions(question, context)

    # Checking if the actual answer matches the expected answer
    assert actual_answer == expected_answer, f'Expected {expected_answer} but got {actual_answer}'

    print('Test passed!')
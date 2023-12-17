# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline, RobertaForQuestionAnswering, RobertaTokenizer

# function_code --------------------

def ask_covid_question(question, context):
    """
    Answers questions about COVID-19 using a pre-trained model.

    Parameters:
        question (str): The question to answer.
        context (str): The context string containing information related to the question.

    Returns:
        dict: A dictionary containing the answer and its details.
    """
    qa_pipeline = pipeline(
        'question-answering', 
        model=RobertaForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2-covid'), 
        tokenizer=RobertaTokenizer.from_pretrained('deepset/roberta-base-squad2-covid')
    )
    return qa_pipeline({'question': question, 'context': context})

# test_function_code --------------------

def test_ask_covid_question():
    print("Testing started.")

    # Test case 1: Common symptoms of COVID-19
    question1 = "What are the common symptoms of COVID-19?"
    context1 = "COVID-19 symptoms include fever, cough, and difficulty breathing."
    answer1 = ask_covid_question(question1, context1)
    assert 'fever' in answer1['answer'], f"Test case 1 failed: {answer1}"

    # Test case 2: COVID-19 transmission
    question2 = "How is COVID-19 transmitted?"
    context2 = "COVID-19 spreads through respiratory droplets from coughs and sneezes."
    answer2 = ask_covid_question(question2, context2)
    assert 'respiratory droplets' in answer2['answer'], f"Test case 2 failed: {answer2}"

    # Test case 3: Effectiveness of masks
    question3 = "Are masks effective in preventing COVID-19?"
    context3 = "Masks can help prevent the spread of the virus from the wearer to others."
    answer3 = ask_covid_question(question3, context3)
    assert 'prevent' in answer3['answer'], f"Test case 3 failed: {answer3}"

    print("Testing finished.")
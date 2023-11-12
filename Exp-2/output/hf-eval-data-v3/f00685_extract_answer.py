# function_import --------------------

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# function_code --------------------

def extract_answer(question: str, context: str) -> str:
    """
    Extracts the answer to a given question from a given context using a pre-trained DeBERTa-v3 model.

    Args:
        question (str): The question to answer.
        context (str): The context from which to extract the answer.

    Returns:
        str: The extracted answer.
    """
    model_name = 'deepset/deberta-v3-large-squad2'
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    QA_input = {
        'question': question,
        'context': context
    }
    answer = nlp(QA_input)
    return answer['answer']

# test_function_code --------------------

def test_extract_answer():
    """
    Tests the extract_answer function.
    """
    question1 = 'What are the benefits of exercise?'
    context1 = 'Exercise helps maintain a healthy body weight, improves cardiovascular health, and boosts the immune system.'
    assert extract_answer(question1, context1) == 'maintain a healthy body weight, improves cardiovascular health, and boosts the immune system'

    question2 = 'Why is model conversion important?'
    context2 = 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
    assert extract_answer(question2, context2) == 'gives freedom to the user and let people easily switch between frameworks'

    question3 = 'What is the task of Question Answering?'
    context3 = 'This is the deberta-v3-large model, fine-tuned using the SQuAD2.0 dataset. It has been trained on question-answer pairs, including unanswerable questions, for the task of Question Answering.'
    assert extract_answer(question3, context3) == 'trained on question-answer pairs, including unanswerable questions'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_answer()
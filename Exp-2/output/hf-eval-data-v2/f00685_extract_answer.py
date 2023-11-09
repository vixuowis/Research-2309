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
    question = 'What are the benefits of exercise?'
    context = 'Exercise helps maintain a healthy body weight, improves cardiovascular health, and boosts the immune system.'
    answer = extract_answer(question, context)
    assert isinstance(answer, str), 'The answer must be a string.'
    assert answer != '', 'The answer must not be an empty string.'

# call_test_function_code --------------------

test_extract_answer()
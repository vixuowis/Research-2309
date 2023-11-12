# function_import --------------------

from transformers import pipeline, AutoModel, AutoTokenizer

# function_code --------------------

def question_answering(question: str, context: str) -> str:
    '''
    This function uses the bert-large-uncased-whole-word-masking-squad2 model from the Transformers library to answer a question based on a given context.

    Args:
        question (str): The question to be answered.
        context (str): The context in which the question is to be answered.

    Returns:
        str: The answer to the question.
    '''
    model = AutoModel.from_pretrained('deepset/bert-large-uncased-whole-word-masking-squad2')
    tokenizer = AutoTokenizer.from_pretrained('deepset/bert-large-uncased-whole-word-masking-squad2')
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    QA_input = {
        'question': question,
        'context': context
    }
    res = nlp(QA_input)
    return res['answer']

# test_function_code --------------------

def test_question_answering():
    '''
    This function tests the question_answering function.
    '''
    assert question_answering('Why is model conversion important?', 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.') == 'Model conversion is important because it gives freedom to the user and allows people to easily switch between different frameworks.'
    assert question_answering('What is the capital of France?', 'Paris is the capital of France.') == 'Paris'
    assert question_answering('Who won the world cup in 2018?', 'The 2018 FIFA World Cup was won by France.') == 'France'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_question_answering()
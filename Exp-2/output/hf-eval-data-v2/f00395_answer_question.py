# function_import --------------------

from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# function_code --------------------

def answer_question(question: str, context: str) -> str:
    """
    This function uses the pre-trained 'deepset/deberta-v3-large-squad2' model from Hugging Face Transformers
    to answer a given question based on the provided context.

    Args:
        question (str): The question to be answered.
        context (str): The context in which the question is asked.

    Returns:
        str: The answer to the question.
    """
    model = AutoModelForQuestionAnswering.from_pretrained('deepset/deberta-v3-large-squad2')
    tokenizer = AutoTokenizer.from_pretrained('deepset/deberta-v3-large-squad2')

    inputs = tokenizer(question, context, return_tensors='pt', max_length=512, truncation=True)
    output = model(**inputs)
    answer_start = output.start_logits.argmax().item()
    answer_end = output.end_logits.argmax().item()

    ans = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end+1]))
    return ans

# test_function_code --------------------

def test_answer_question():
    """
    This function tests the 'answer_question' function with a sample question and context.
    """
    question = 'Why is model conversion important?'
    context = 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
    answer = answer_question(question, context)
    assert isinstance(answer, str), 'The function should return a string.'
    assert len(answer) > 0, 'The answer should not be an empty string.'

# call_test_function_code --------------------

test_answer_question()
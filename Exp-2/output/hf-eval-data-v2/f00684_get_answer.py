# function_import --------------------

from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# function_code --------------------

def get_answer(question: str, context: str) -> str:
    """
    This function takes a question and a context as input and returns the answer to the question based on the context.
    The function uses the pre-trained ELECTRA_large_discriminator language model fine-tuned on the SQuAD2.0 dataset.

    Args:
        question (str): The question to be answered.
        context (str): The context in which to find the answer.

    Returns:
        str: The answer to the question based on the context.
    """
    model = AutoModelForQuestionAnswering.from_pretrained('ahotrod/electra_large_discriminator_squad2_512')
    tokenizer = AutoTokenizer.from_pretrained('ahotrod/electra_large_discriminator_squad2_512')
    inputs = tokenizer(question, context, return_tensors='pt')
    outputs = model(**inputs)
    answer_start = outputs.start_logits.argmax().item()
    answer_end = outputs.end_logits.argmax().item() + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    return answer

# test_function_code --------------------

def test_get_answer():
    """
    This function tests the get_answer function.
    It uses a sample question and context and checks if the returned answer is correct.
    """
    question = 'What is the capital of France?'
    context = 'France is a country in Europe. Its capital is Paris.'
    expected_answer = 'Paris'
    assert get_answer(question, context) == expected_answer

# call_test_function_code --------------------

test_get_answer()
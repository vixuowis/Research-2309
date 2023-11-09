# function_import --------------------

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# function_code --------------------

def get_answer(question: str, text: str) -> str:
    """
    This function uses the 'valhalla/longformer-base-4096-finetuned-squadv1' model to answer a question based on the provided text.

    Args:
        question (str): The question to be answered.
        text (str): The text from which the answer will be extracted.

    Returns:
        str: The answer to the question.
    """
    tokenizer = AutoTokenizer.from_pretrained('valhalla/longformer-base-4096-finetuned-squadv1')
    model = AutoModelForQuestionAnswering.from_pretrained('valhalla/longformer-base-4096-finetuned-squadv1')
    encoding = tokenizer(question, text, return_tensors='pt')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    start_scores, end_scores = model(input_ids, attention_mask=attention_mask)
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    answer_tokens = all_tokens[torch.argmax(start_scores) :torch.argmax(end_scores)+1]
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))

# test_function_code --------------------

def test_get_answer():
    """
    This function tests the 'get_answer' function.
    """
    question = 'What has Huggingface done ?'
    text = 'Huggingface has democratized NLP. Huge thanks to Huggingface for this.'
    assert isinstance(get_answer(question, text), str)

# call_test_function_code --------------------

test_get_answer()
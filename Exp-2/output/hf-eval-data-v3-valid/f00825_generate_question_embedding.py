# function_import --------------------

import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

# function_code --------------------

def generate_question_embedding(question):
    """
    Generate question embedding using DPRQuestionEncoder.

    Args:
        question (str): The question to be encoded.

    Returns:
        torch.Tensor: The question embedding.
    """
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

    input_ids = tokenizer(question, return_tensors='pt')['input_ids']
    question_embedding = model(input_ids).pooler_output
    return question_embedding

# test_function_code --------------------

def test_generate_question_embedding():
    """
    Test the function generate_question_embedding.
    """
    question = 'What are the best attractions in Paris?'
    embedding = generate_question_embedding(question)
    assert embedding is not None
    assert embedding.size() == torch.Size([1, 768])

    question = 'What is the capital of France?'
    embedding = generate_question_embedding(question)
    assert embedding is not None
    assert embedding.size() == torch.Size([1, 768])

    question = 'Who is the president of the United States?'
    embedding = generate_question_embedding(question)
    assert embedding is not None
    assert embedding.size() == torch.Size([1, 768])

    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_generate_question_embedding())
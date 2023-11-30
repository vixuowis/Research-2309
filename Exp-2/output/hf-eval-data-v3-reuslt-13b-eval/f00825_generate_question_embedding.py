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
    
    # Initialize question encoder and tokenizer.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")
    
    # Move the model to target device (cuda or cpu).
    model.to(device)

    input_ids = tokenizer([question], return_tensors="pt", padding=True, truncation=True).to(device)["input_ids"]
    attention_mask = torch.where(input_ids != 0, 1, 0)

    # Get question embedding.
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    return outputs["pooler_output"].detach().cpu()

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
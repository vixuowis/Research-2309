# requirements_file --------------------

!pip install -U transformers  torch

# function_import --------------------

from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

# function_code --------------------

def answer_question_with_deberta(question, context):
    """
    This function takes a question and a context as input and uses the DeBERTa v3
    large model to find the answer within the context.
    
    Args:
        question (str): The question to be answered.
        context (str): The context containing the information to answer the question.
    
    Returns:
        str: The answer to the question.
    """
    # Load the pre-trained model and tokenizer
    model_name = 'deepset/deberta-v3-large-squad2'
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize the inputs
    inputs = tokenizer(question, context, return_tensors='pt', max_length=512, truncation=True)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Find the tokens with the highest `start` and `end` logit scores
    answer_start = outputs.start_logits.argmax().item()
    answer_end = outputs.end_logits.argmax().item() + 1
    
    # Convert the tokens to a string
    answer_tokens = inputs['input_ids'][0][answer_start:answer_end]
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(answer_tokens))
    
    return answer

# test_function_code --------------------

def test_answer_question_with_deberta():
    print("Testing started.")
    
    # Define a question and context.
    question = 'Who was the founding father of quantum mechanics?'
    context = 'Niels Bohr was a Danish physicist who made foundational contributions to understanding atomic structure and quantum theory, for which he received the Nobel Prize in Physics in 1922. Bohr was one of the founding fathers of quantum mechanics.'
    
    # Test the function
    print("Testing case started.")
    answer = answer_question_with_deberta(question, context)
    assert answer == 'Niels Bohr', f"Test case failed: Expected 'Niels Bohr', got {answer}"  
    print("Testing finished.")
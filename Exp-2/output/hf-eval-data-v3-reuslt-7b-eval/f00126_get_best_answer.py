# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# function_code --------------------

def get_best_answer(query: str, passages: list) -> str:
    """
    Given a query and a list of passages, this function returns the passage that best answers the query.
    
    Args:
        query (str): The question to be answered.
        passages (list): A list of possible answer passages.
    
    Returns:
        str: The passage that best answers the query.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    
    # Tokenize query and passages, then add them to the tokenizer
    inputs = tokenizer(["[CLS] " + query + " [SEP]"], 
                       text_pair=passages, 
                       padding="max_length", 
                       max_length=512, 
                       truncation=True)
    
    # Load the model into CUDA if available, otherwise load to CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Convert the input ids to a PyTorch tensor and add it as an argument to the forward method.
    inputs = {k:torch.tensor([v], device=device) for (k, v) in inputs.items()} 
    
    outputs = model(**inputs)   # Perform inference on the input
    logits = outputs[0]         # Grab the scores produced by the first head of our base model
    pred_score, pred_index = torch.max(logits, dim=1)  # Get the highest prediction score and index
    
    return passages[int(pred_index)]


# test_function_code --------------------

def test_get_best_answer():
    """
    Test the function get_best_answer.
    """
    query = 'What is the capital of France?'
    passages = ['Paris is the capital of France.', 'London is the capital of England.', 'Berlin is the capital of Germany.']
    assert get_best_answer(query, passages) == 'Paris is the capital of France.'
    
    query = 'Who won the world cup in 2018?'
    passages = ['France won the world cup in 2018.', 'Germany won the world cup in 2014.', 'Brazil won the world cup in 2002.']
    assert get_best_answer(query, passages) == 'France won the world cup in 2018.'
    
    query = 'Who is the CEO of Tesla?'
    passages = ['Elon Musk is the CEO of Tesla.', 'Bill Gates is the CEO of Microsoft.', 'Jeff Bezos is the CEO of Amazon.']
    assert get_best_answer(query, passages) == 'Elon Musk is the CEO of Tesla.'
    
    return 'All Tests Passed'


# call_test_function_code --------------------

test_get_best_answer()
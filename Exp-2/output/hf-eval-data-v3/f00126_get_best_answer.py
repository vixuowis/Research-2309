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
    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-2-v2')
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-2-v2')
    features = tokenizer([query]*len(passages), passages, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        scores = model(**features).logits
    sorted_passages = [passages[idx] for idx in scores.argsort(descending=True)]
    best_passage = sorted_passages[0]
    return best_passage

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
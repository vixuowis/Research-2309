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
    
    # Initialize the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-answering-squadv2")
    model = AutoModelForSequenceClassification.from_pretrained("mrm8488/t5-base-finetuned-question-answering-squadv2").to(0)
    
    # Tokenize the query and all passages
    inputs = tokenizer([[query + ' ' + passage for passage in passages]], return_tensors='pt', max_length=384, padding=True).to(0)
    outputs = model(**inputs)[0][:, 1]
    
    # Get the index of the best answer
    index = torch.argmax(outputs)
    
    return passages[index]

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
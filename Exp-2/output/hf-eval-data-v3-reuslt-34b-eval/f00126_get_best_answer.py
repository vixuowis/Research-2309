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

    # load tokenizer and model from pretrained weights
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english', return_dict=False)
    model.to(device)
    
    # prepare inputs from query and passages (assumes that there are 4 passages in the list)
    passage_inputs = [[query, p] for p in passages[:4]]
    encoded_passage_inputs = [tokenizer(*p, truncation=True) for p in passage_inputs]
    
    # prepare input tensors to feed into model
    max_len = min(max([len(input['input_ids']) for input in encoded_passage_inputs]), 512)
    for i, p in enumerate(encoded_passage_inputs):
        encoded_passage_inputs[i]['input_ids'] = encoded_passage_inputs[i]['input_ids'][:max_len]
        encoded_passage_inputs[i]['attention_mask'] = encoded_passage_inputs[i]['attention_mask'][:max_len]
    passage_inputs_tensors = [torch.tensor(p).to(device) for p in encoded_passage_inputs]
    
    # feed passages through model to get a logit score and select the one with the highest score
    scores = []
    for input_ids, attention_mask in zip(passage_inputs_tensors[0], passage_inputs_tensors[1]):
        outputs = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
        scores.append((float(outputs[0][0]), passages[i]))
    best_passage = max(scores, key=lambda x: x[0])[1]


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
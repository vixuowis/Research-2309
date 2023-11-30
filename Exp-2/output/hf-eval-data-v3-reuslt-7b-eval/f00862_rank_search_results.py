# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# function_code --------------------

def rank_search_results(query: str, passages: list) -> list:
    """
    Ranks the given passages based on their relevance to the given query using a pretrained model.

    Args:
        query (str): The search query.
        passages (list): The list of passages to be ranked.

    Returns:
        list: The list of passages ranked in descending order of relevance.
    """
    
    if not passages: # empty list check
        return []
        
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad", 
                                                               return_dict=True)
    
    # Preprocess the list of passages to a single string: passage_str
    passage_str = ""
    for passage in passages:
        passage_str += " ".join(passage) + " "
        
    # Tokenize and encode the query and passage.
    inputs = tokenizer("[CLS] " + query + " [SEP]"  + passage_str + "[SEP]", return_tensors="pt")
    
    with torch.no_grad():
        # Get the logits for each example.
        outputs = model(**inputs)
        
    # get the scores for each example
    logits = outputs.logits[:, 0]
    
    # Rank in descending order.
    _, indices = torch.sort(-1 * logits)
    
    return [passages[i] for i in indices]

# test_function_code --------------------

def test_rank_search_results():
    """
    Tests the rank_search_results function with some test cases.
    """
    query = 'How many people live in Berlin?'
    passages = [
        'Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.',
        'New York City is famous for the Metropolitan Museum of Art.',
        'Berlin is the capital of Germany and one of the 16 states of Germany.',
        'Berlin is known for its festivals, diverse architecture, nightlife, contemporary arts, and a high quality of living.'
    ]
    result = rank_search_results(query, passages)
    assert isinstance(result, list), 'The result should be a list.'
    assert len(result) == len(passages), 'The result should have the same length as the input passages.'
    assert all(isinstance(item, tuple) and len(item) == 2 for item in result), 'Each item in the result should be a tuple with two elements.'
    assert all(isinstance(item[0], str) and isinstance(item[1], float) for item in result), 'Each item in the result should be a tuple with a string and a float.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_rank_search_results()
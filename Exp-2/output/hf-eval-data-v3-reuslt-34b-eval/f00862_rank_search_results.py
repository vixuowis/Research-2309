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

    # Load tokenizer and model from Huggingface hub
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "mrm8488/bert-mini-finetuned-age_news-classification"
    )

    # Pair each passage with the query to be fed into the model
    inputs = [[passage, query] for passage in passages]

    # Tokenize the pair of passage and query
    tokenized_inputs = [tokenizer(pair) for pair in inputs]

    # Extract input ids from the tokenized inputs and convert into a PyTorch tensor
    input_ids = torch.tensor([pair["input_ids"] for pair in tokenized_inputs])

    # Feed passage-query pairs to the model and get their corresponding probabilities of being relevant or not
    outputs = model(input_ids)
    logits = outputs[0]
    relevance_prob = torch.softmax(logits[:, 1], dim=0)
    relevance_prob = list(relevance_prob.detach().numpy())

    # Sort the passages in descending order of their relevance probabilities to the query
    ranked_passages = [
        passage for _, passage in sorted(zip(relevance_prob, passages), reverse=True)
    ]

    return ranked_passages

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
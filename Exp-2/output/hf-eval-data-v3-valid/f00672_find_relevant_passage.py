# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# function_code --------------------

def find_relevant_passage(question: str, candidate_passages: list) -> str:
    """
    Find the most relevant passage given a question and several candidate passages.

    Args:
        question (str): The question to be answered.
        candidate_passages (list): A list of candidate passages.

    Returns:
        str: The most relevant passage.

    Raises:
        OSError: If there is a problem with loading the pre-trained model or tokenizer.
    """
    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    features = tokenizer([question] * len(candidate_passages), candidate_passages, padding=True, truncation=True, return_tensors='pt')
    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
        sorted_passages = [x for _, x in sorted(zip(scores.detach().numpy(), candidate_passages), reverse=True)]
    return sorted_passages[0]

# test_function_code --------------------

def test_find_relevant_passage():
    """
    Test the function find_relevant_passage.
    """
    question = 'How many people live in Berlin?'
    candidate_passages = ['Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 'New York City is famous for the Metropolitan Museum of Art.']
    assert isinstance(find_relevant_passage(question, candidate_passages), str)
    question = 'What is the capital of Germany?'
    candidate_passages = ['Berlin is the capital of Germany.', 'Paris is the capital of France.']
    assert find_relevant_passage(question, candidate_passages) == 'Berlin is the capital of Germany.'
    question = 'Who won the world cup in 2014?'
    candidate_passages = ['Germany won the world cup in 2014.', 'Brazil hosted the world cup in 2014.']
    assert find_relevant_passage(question, candidate_passages) == 'Germany won the world cup in 2014.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_find_relevant_passage()
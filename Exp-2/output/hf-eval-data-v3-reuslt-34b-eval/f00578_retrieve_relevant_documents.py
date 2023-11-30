# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# function_code --------------------

def retrieve_relevant_documents(query: str, documents: list) -> list:
    """
    Retrieve relevant documents based on a user's query using Hugging Face Transformers.

    Args:
        query (str): The user's query.
        documents (list): A list of documents to retrieve information from.

    Returns:
        list: A list of documents sorted based on their relevance to the query.
    """
    
    # Load model and tokenizer - https://huggingface.co/bert-base-uncased
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Encode query and documents
    input_ids = torch.tensor([tokenizer.encode(query, d) for d in documents])
    
    # Retrieve the probability of each document being relevant to the query
    probs = [torch.softmax(model(input_ids[i:i+1])['logits'], dim=1)[0][1] \
             for i in range(len(documents))]
    
    # Sort documents based on their probability of being relevant to the query
    probs, documents = zip(*sorted(zip(probs, documents), reverse=True))
    
    return list(documents)

# test_function_code --------------------

def test_retrieve_relevant_documents():
    query = 'How many people live in Berlin?'
    documents = ['Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 'New York City is famous for the Metropolitan Museum of Art.']
    expected_output = ['Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 'New York City is famous for the Metropolitan Museum of Art.']
    assert retrieve_relevant_documents(query, documents) == expected_output

    query = 'What is the capital of Germany?'
    documents = ['Berlin is the capital of Germany.', 'Paris is the capital of France.']
    expected_output = ['Berlin is the capital of Germany.', 'Paris is the capital of France.']
    assert retrieve_relevant_documents(query, documents) == expected_output

    query = 'What is the population of New York City?'
    documents = ['New York City has a population of 8,398,748 people.', 'Los Angeles has a population of 3,979,576 people.']
    expected_output = ['New York City has a population of 8,398,748 people.', 'Los Angeles has a population of 3,979,576 people.']
    assert retrieve_relevant_documents(query, documents) == expected_output

    return 'All Tests Passed'


# call_test_function_code --------------------

test_retrieve_relevant_documents()
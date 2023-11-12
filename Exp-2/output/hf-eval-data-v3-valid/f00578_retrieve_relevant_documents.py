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
    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-2-v2')
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-2-v2')

    features = tokenizer([query]*len(documents), documents, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        scores = model(**features).logits
    sorted_docs = [doc for _, doc in sorted(zip(scores, documents), reverse=True)]
    return sorted_docs

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
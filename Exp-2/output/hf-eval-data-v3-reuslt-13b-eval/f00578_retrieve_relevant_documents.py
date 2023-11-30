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

    # Load Hugging Face model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-cased-distilled-squad', return_dict=True)

    # Tokenize the query and the documents.
    inputs = tokenizer(query, documents, truncation=True, padding='max_length')
    features = {key: torch.as_tensor([val]) for key, val in inputs.items()}

    # Get predictions on the query/documents pair.
    with torch.no_grad():
        output = model(**features)
    
    scores = [item.logits for item in output]
    scores = [scores[0][i].item() for i in range(len(documents))]

    # Sort the documents by their relevance to the query.
    sorted_documents, _ = list(zip(*sorted(list(zip(documents, scores)), key=lambda x: -x[1])))
    return list(sorted_documents)

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
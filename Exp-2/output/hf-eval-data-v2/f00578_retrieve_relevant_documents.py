# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# function_code --------------------

def retrieve_relevant_documents(query, documents):
    """
    Retrieves the most relevant documents based on a user's query using the Hugging Face Transformers model.

    Args:
        query (str): The user's query.
        documents (list): A list of documents to search from.

    Returns:
        list: A list of documents sorted based on their relevance to the query in descending order.
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
    """
    Tests the retrieve_relevant_documents function.
    """
    query = 'How many people live in Berlin?'
    documents = ['Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 'New York City is famous for the Metropolitan Museum of Art.']
    expected_output = ['Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 'New York City is famous for the Metropolitan Museum of Art.']
    assert retrieve_relevant_documents(query, documents) == expected_output

# call_test_function_code --------------------

test_retrieve_relevant_documents()
# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# function_code --------------------

def retrieve_documents(query, documents):
    """
    Retrieve relevant documents based on a user's query from a collection of documents.

    Args:
        query (str): The user's query for which relevant documents need to be identified.
        documents (list of str): A list of documents from which relevant information is to be retrieved.

    Returns:
        list of str: A sorted list of documents ranked by their relevance to the query.

    Raises:
        ValueError: If the `documents` list is empty.
    """
    if not documents:
        raise ValueError('The document list is empty.')

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-2-v2')
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-2-v2')

    # Tokenize and prepare features
    features = tokenizer([query]*len(documents), documents, padding=True, truncation=True, return_tensors='pt')

    # Calculate relevance scores
    with torch.no_grad():
        scores = model(**features).logits

    # Sort documents based on scores
    sorted_docs = [doc for _, doc in sorted(zip(scores.squeeze(), documents), key=lambda pair: pair[0], reverse=True)]

    return sorted_docs

# test_function_code --------------------

def test_retrieve_documents():
    print('Testing started.')
    query = 'How many people live in Berlin?'
    documents = [
        'Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.',
        'New York City is famous for the Metropolitan Museum of Art.',
        'Berlin is known for its historical associations as the German capital.'
    ]

    # Test case 1: Regular input
    print('Testing case [1/3] started.')
    retrieved_docs = retrieve_documents(query, documents)
    assert retrieved_docs[0].startswith('Berlin has a population'), 'Test case [1/3] failed: First document should be the most relevant.'

    # Test case 2: Empty document list
    print('Testing case [2/3] started.')
    try:
        retrieve_documents(query, [])
        assert False, 'Test case [2/3] failed: Should have raised a ValueError for empty document list.'
    except ValueError:
        pass

    # Test case 3: Query unrelated to documents
    print('Testing case [3/3] started.')
    unrelated_query = 'What is the capital of France?'
    retrieved_docs_unrelated = retrieve_documents(unrelated_query, documents)
    assert not any('Berlin' in doc for doc in retrieved_docs_unrelated), 'Test case [3/3] failed: Documents should not be relevant to the unrelated query.'
    print('Testing finished.')


# call_test_function_line --------------------

test_retrieve_documents()
# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# function_code --------------------

def rank_search_results(query, documents):
    """
    This function ranks given list of documents based on their relevance to a search query.
    :param query: str, The search query.
    :param documents: list of str, The documents to be ranked.
    :return: list of tuples, Sorted documents with their relevance scores (highest first).
    """
    # Load the pre-trained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Tokenize and encode the query with each document
    features = tokenizer([query] * len(documents), documents, padding=True, truncation=True, return_tensors='pt')

    # Evaluate the model to get relevance scores
    model.eval()
    with torch.no_grad():
        scores = model(**features).logits

    # Sort the documents by their scores in descending order
    sorted_docs = sorted(zip(documents, scores.squeeze().tolist()), key=lambda x: x[1], reverse=True)
    return sorted_docs

# test_function_code --------------------

def test_rank_search_results():
    print('Testing rank_search_results function.')
    test_query = 'How many people live in Berlin?'
    test_documents = [
        'Berlin has a population of 3,520,031 registered inhabitants.',
        'New York City is famous for the Metropolitan Museum of Art.',
        'Berlin is notable for its historical associations as the German capital.'
    ]

    expected_result = test_documents[:1] + test_documents[2:]  # Assuming the first and last documents are more relevant
    ranked_docs = rank_search_results(test_query, test_documents)
    assert [doc for doc, _ in ranked_docs] == expected_result, 'Test failed: The documents were not ranked correctly.'
    print('Test passed.')

test_rank_search_results()
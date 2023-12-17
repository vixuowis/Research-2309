# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# function_code --------------------

def retrieve_relevant_documents(query, documents):
    """
    Given a query and a list of documents, this function retrieves the most relevant documents.
    It uses the 'cross-encoder/ms-marco-TinyBERT-L-2-v2' provided by Hugging Face Transformers.

    Parameters:
        query (str): The user query to evaluate.
        documents (list): A list of documents to compare against the query.

    Returns:
        list: The list of documents ordered by relevance to the query.
    """
    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-2-v2')
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-2-v2')
    features = tokenizer([query]*len(documents), documents, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        scores = model(**features).logits
    sorted_docs = [doc for _, doc in sorted(zip(scores.tolist(), documents), key=lambda x: x[0], reverse=True)]
    return sorted_docs

# test_function_code --------------------

def test_retrieve_relevant_documents():
    print("Testing started.")
    query = "What causes rainfall?"
    documents = [
        "Precipitation is any kind of water that falls from the atmosphere, such as rain, snow, sleet, and hail.",
        "Rainfall is caused by the condensation of moisture in the air.",
        "The water cycle is the process of evaporation, condensation, and precipitation."]

    expected_result = [
        "Rainfall is caused by the condensation of moisture in the air.",
        "The water cycle is the process of evaporation, condensation, and precipitation.",
        "Precipitation is any kind of water that falls from the atmosphere, such as rain, snow, sleet, and hail."]

    print("Testing single query and multiple documents scenario.")
    retrieved_documents = retrieve_relevant_documents(query, documents)
    assert retrieved_documents == expected_result, f"Test failed: Retrieved documents do not match the expected result."
    print("Test passed: Single query and multiple documents scenario.")

    print("Testing finished.")

# Running the test function
test_retrieve_relevant_documents()
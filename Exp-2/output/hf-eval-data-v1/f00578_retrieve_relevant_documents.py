from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def retrieve_relevant_documents(query, documents):
    '''
    This function retrieves the most relevant documents based on a user's query using the Hugging Face Transformers model 'cross-encoder/ms-marco-TinyBERT-L-2-v2'.
    
    Parameters:
    query (str): The user's query.
    documents (list): The collection of documents.
    
    Returns:
    list: The sorted list of documents in decreasing order of relevance.
    '''
    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-2-v2')
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-2-v2')
    
    features = tokenizer([query]*len(documents), documents, padding=True, truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        scores = model(**features).logits
    
    sorted_docs = [doc for _, doc in sorted(zip(scores, documents), reverse=True)]
    
    return sorted_docs
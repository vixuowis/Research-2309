from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch


def calculate_review_similarity(review1: str, review2: str) -> float:
    """
    This function calculates the similarity between two book reviews using a pretrained model from Hugging Face Transformers.
    The model used is 'princeton-nlp/unsup-simcse-roberta-base'.
    The similarity is calculated using cosine similarity.
    
    Parameters:
    review1 (str): The first book review.
    review2 (str): The second book review.
    
    Returns:
    float: The similarity score between the two reviews. The score is in the range of [-1, 1], with higher scores indicating more similarity.
    """
    # Load the pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/unsup-simcse-roberta-base')
    model = AutoModel.from_pretrained('princeton-nlp/unsup-simcse-roberta-base')
    
    # Tokenize and convert the reviews into input tensors
    input_tensors = tokenizer([review1, review2], return_tensors='pt', padding=True, truncation=True)
    
    # Pass the tensors to the model to get sentence embeddings
    embeddings = model(**input_tensors).pooler_output
    
    # Calculate the similarity between the embeddings
    similarity_score = cosine_similarity(embeddings[0].detach().numpy().reshape(1, -1), embeddings[1].detach().numpy().reshape(1, -1))[0][0]
    
    return similarity_score
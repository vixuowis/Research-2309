from transformers import AutoTokenizer, AutoModel
import torch

# Function to calculate semantic similarity

def calculate_semantic_similarity(text1, text2):
    """
    This function calculates the semantic similarity between two texts using the pretrained 'princeton-nlp/sup-simcse-roberta-large' model.
    
    Parameters:
    text1 (str): The first text
    text2 (str): The second text
    
    Returns:
    float: The semantic similarity between the two texts
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-roberta-large')
    model = AutoModel.from_pretrained('princeton-nlp/sup-simcse-roberta-large')
    
    # Tokenize the texts
    inputs1 = tokenizer(text1, return_tensors='pt')
    inputs2 = tokenizer(text2, return_tensors='pt')
    
    # Get the embeddings
    embeddings1 = model(**inputs1)
    embeddings2 = model(**inputs2)
    
    # Calculate the cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(embeddings1[0], embeddings2[0])
    
    return cos_sim.item()
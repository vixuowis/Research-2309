from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

def find_similar_sentence(source_sentence, sentences_to_compare):
    """
    This function finds the most similar sentence to the source sentence from a list of sentences.
    It uses the 'GanymedeNil/text2vec-large-chinese' model from Hugging Face Transformers to create embeddings for the sentences and calculates the cosine similarity between them.
    
    Parameters:
    source_sentence (str): The source sentence to which we want to find a similar sentence.
    sentences_to_compare (list): The list of sentences to compare with the source sentence.
    
    Returns:
    str: The most similar sentence to the source sentence.
    """
    tokenizer = AutoTokenizer.from_pretrained('GanymedeNil/text2vec-large-chinese')
    model = AutoModel.from_pretrained('GanymedeNil/text2vec-large-chinese')
    
    def encode(sentence):
        input_ids = tokenizer(sentence, return_tensors="pt").input_ids
        return model(input_ids).last_hidden_state.mean(1).detach()
    
    source_embedding = encode(source_sentence)
    sentence_embeddings = torch.stack([encode(candidate) for candidate in sentences_to_compare])
    
    similarity_scores = cosine_similarity(source_embedding.cpu(), sentence_embeddings.cpu())
    highest_similarity_index = similarity_scores.argmax()
    
    return sentences_to_compare[highest_similarity_index]
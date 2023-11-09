from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def recommend_articles(user_liked_articles, new_articles, threshold=0.8):
    """
    This function recommends articles to users based on how similar the articles are to their previously liked articles.
    It uses the 'princeton-nlp/unsup-simcse-roberta-base' model from Hugging Face for calculating sentence similarity.
    
    Parameters:
    user_liked_articles (list): A list of texts from articles the user has previously liked.
    new_articles (list): A list of texts from new articles that are to be recommended.
    threshold (float): The similarity threshold for recommending an article. Default is 0.8.
    
    Returns:
    recommended_articles (list): A list of recommended articles.
    """
    tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/unsup-simcse-roberta-base')
    model = AutoModel.from_pretrained('princeton-nlp/unsup-simcse-roberta-base')
    
    # Compute sentence embeddings for user liked articles
    liked_embeddings = []
    for article in user_liked_articles:
        inputs = tokenizer(article, return_tensors='pt')
        outputs = model(**inputs)
        liked_embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
    
    # Compute sentence embeddings for new articles
    new_embeddings = []
    for article in new_articles:
        inputs = tokenizer(article, return_tensors='pt')
        outputs = model(**inputs)
        new_embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
    
    # Compute similarity and recommend articles
    recommended_articles = []
    for i, new_emb in enumerate(new_embeddings):
        for liked_emb in liked_embeddings:
            similarity = cosine_similarity(new_emb, liked_emb)
            if np.mean(similarity) > threshold:
                recommended_articles.append(new_articles[i])
                break
    
    return recommended_articles
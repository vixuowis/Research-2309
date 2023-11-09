from transformers import pipeline

def toxic_comment_classifier(comment):
    '''
    This function uses the 'martin-ha/toxic-comment-model' from Hugging Face Transformers to classify a comment as toxic or not.
    The model is a fine-tuned version of the DistilBERT model for toxic comment classification.
    
    Parameters:
    comment (str): The comment to be classified.
    
    Returns:
    float: The toxicity score of the comment.
    '''
    # Initialize the toxic comment classifier
    toxic_classifier = pipeline(model='martin-ha/toxic-comment-model')
    
    # Get the toxicity score of the comment
    toxicity_score = toxic_classifier(comment)
    
    return toxicity_score
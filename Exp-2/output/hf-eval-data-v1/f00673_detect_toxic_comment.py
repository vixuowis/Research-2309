from transformers import pipeline

def detect_toxic_comment(comment):
    """
    This function uses a pre-trained model from Hugging Face Transformers to classify a comment as toxic or not.
    The model used is 'martin-ha/toxic-comment-model', a fine-tuned DistilBERT model designed for detecting toxic comments.
    
    Parameters:
    comment (str): The comment to be classified.
    
    Returns:
    dict: The classification result from the model.
    """
    # Create a pipeline with the specified model
    toxic_comment_detector = pipeline(model='martin-ha/toxic-comment-model')
    
    # Classify the comment
    toxicity_classification = toxic_comment_detector(comment)
    
    return toxicity_classification
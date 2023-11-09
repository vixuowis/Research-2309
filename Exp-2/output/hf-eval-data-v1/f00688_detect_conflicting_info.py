from sentence_transformers import CrossEncoder


def detect_conflicting_info(sentence_pairs):
    """
    This function uses the 'cross-encoder/nli-MiniLM2-L6-H768' model from Hugging Face Transformers to detect if a given summary of a book contains conflicting information.
    It creates a list of sentence pairs from the summary and the model predicts the probability scores for each sentence pair belonging to contradiction, entailment, or neutral categories.
    High contradiction scores for a pair of sentences indicate conflicting information.
    
    Args:
    sentence_pairs (list): A list of tuples where each tuple contains two sentences from the summary.
    
    Returns:
    scores (list): A list of scores where each score corresponds to the probability of contradiction, entailment, or neutral for each sentence pair.
    """
    model = CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')
    scores = model.predict(sentence_pairs)
    return scores
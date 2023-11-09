from transformers import pipeline

def fill_mask_french(sentence):
    """
    This function uses the CamemBERT model from Hugging Face's transformers library to fill in the blanks in French sentences.
    The model is a state-of-the-art language model for French based on the RoBERTa model.
    It is available on Hugging Face in 6 different versions with varying number of parameters, amount of pretraining data, and pretraining data source domains.
    It can be used for Fill-Mask tasks.
    
    Parameters:
    sentence (str): The sentence with a masked token ('<mask>') that needs to be filled.
    
    Returns:
    list: A list of dictionaries with the filled sentence and the score of the prediction.
    """
    camembert_fill_mask = pipeline('fill-mask', model='camembert-base', tokenizer='camembert-base')
    results = camembert_fill_mask(sentence)
    return results
from transformers import DebertaV2ForMaskedLM, DebertaV2Tokenizer
import torch

def predict_next_word(phrase):
    """
    This function uses the DebertaV2ForMaskedLM model from the transformers library to predict the next word in a given phrase.
    The model has been pre-trained on a large corpus of text and is capable of predicting the masked word/token in a given context.
    
    Parameters:
    phrase (str): The phrase in which to predict the next word.
    
    Returns:
    str: The predicted next word.
    """
    # Load the pre-trained model
    mask_model = DebertaV2ForMaskedLM.from_pretrained('microsoft/deberta-v2-xxlarge')
    tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v2-xxlarge')
    
    # Prepare the phrase for the model
    processed = tokenizer(phrase, return_tensors='pt')
    
    # Use the model to predict the next word
    predictions = mask_model(**processed).logits.argmax(dim=-1)
    
    # Decode the prediction to get the word
    predicted_word = tokenizer.decode(predictions[0], skip_special_tokens=True)
    
    return predicted_word
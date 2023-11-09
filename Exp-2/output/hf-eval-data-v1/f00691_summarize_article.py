from transformers import T5Tokenizer, T5Model


def summarize_article(article):
    """
    This function uses the T5 large model from Hugging Face Transformers to generate a summary of a lengthy article.
    
    Parameters:
    article (str): The article to be summarized.
    
    Returns:
    str: The summarized article.
    """
    # Initialize the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    model = T5Model.from_pretrained('t5-large')
    
    # Tokenize the article and the prompt
    input_ids = tokenizer('summarize: ' + article, return_tensors='pt').input_ids
    decoder_input_ids = tokenizer('summarize: ', return_tensors='pt').input_ids
    
    # Pass the tokens to the model
    outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    
    # Return the summarized article
    return outputs

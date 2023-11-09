from transformers import AutoTokenizer, AutoModelForMaskedLM


def fill_mask_japanese(masked_text):
    """
    This function fills in the missing words in a given Japanese text.
    It uses the 'cl-tohoku/bert-base-japanese' pretrained model from the transformers package.
    
    Args:
    masked_text (str): The input text with a masked word '[MASK]'.
    
    Returns:
    str: The input text with the masked word replaced by the predicted word.
    """
    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
    model = AutoModelForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese')
    
    # Process the input text
    encoded_input = tokenizer(masked_text, return_tensors='pt')
    
    # Use the model to predict the missing word
    outputs = model(**encoded_input)
    prediction = outputs.logits.argmax(-1)
    
    # Convert the predicted token id to a word
    predicted_token = tokenizer.convert_ids_to_tokens(prediction[0])
    
    # Replace the masked word with the predicted word
    filled_text = masked_text.replace('[MASK]', predicted_token[1])
    
    return filled_text
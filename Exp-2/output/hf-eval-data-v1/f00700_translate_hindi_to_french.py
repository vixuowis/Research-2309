from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


def translate_hindi_to_french(message_hi):
    """
    This function translates a given message from Hindi to French using the Hugging Face's transformers library.
    The translation model used is 'facebook/mbart-large-50-many-to-many-mmt'.
    
    Args:
    message_hi (str): The message in Hindi to be translated.
    
    Returns:
    str: The translated message in French.
    """
    # Load the pre-trained model and tokenizer
    model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
    tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
    
    # Set the source language to Hindi
    tokenizer.src_lang = 'hi_IN'
    
    # Tokenize the input message
    encoded_hi = tokenizer(message_hi, return_tensors='pt')
    
    # Generate the translated message
    generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.lang_code_to_id['fr_XX'])
    
    # Decode the generated tokens to get the translated message
    translated_message = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    return translated_message
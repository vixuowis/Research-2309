from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Function to translate English text to French using Hugging Face's M2M100 model

def translate_english_to_french(english_text):
    '''
    Translates English text to French using Hugging Face's M2M100 model.
    
    Parameters:
    english_text (str): The English text to be translated.
    
    Returns:
    str: The translated French text.
    '''
    # Load the model and tokenizer
    model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')
    tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M')
    
    # Set the source language to English
    tokenizer.src_lang = 'en'
    
    # Encode the English text
    encoded_input = tokenizer(english_text, return_tensors='pt')
    
    # Generate the translated text in French
    generated_tokens = model.generate(**encoded_input, forced_bos_token_id=tokenizer.get_lang_id('fr'))
    
    # Decode the generated tokens to get the French text
    french_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    return french_text
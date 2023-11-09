from transformers import T5Tokenizer, T5ForConditionalGeneration


def translate_english_to_german(input_text):
    """
    This function translates English sentences to German using the Hugging Face Transformers library.
    It uses the 'google/flan-t5-xl' model which is a large-scale language model fine-tuned on more than 1000 tasks covering multiple languages.
    
    Parameters:
    input_text (str): The English sentence to be translated.
    
    Returns:
    str: The translated German sentence.
    """
    # Load the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-xl')
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-xl')
    
    # Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    
    # Generate the translated tokens
    outputs = model.generate(input_ids)
    
    # Decode the translated tokens back into a readable German sentence
    return tokenizer.decode(outputs[0])
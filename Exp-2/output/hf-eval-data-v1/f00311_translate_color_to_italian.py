from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# Function to translate color names from English to Italian
# using the Hugging Face Transformers library

def translate_color_to_italian(color_name):
    """
    This function takes a color name in English and translates it to Italian using the Hugging Face Transformers library.
    
    Parameters:
    color_name (str): The name of the color in English.
    
    Returns:
    str: The name of the color in Italian.
    """
    # Load the pre-trained model and tokenizer
    model = MT5ForConditionalGeneration.from_pretrained('google/mt5-base')
    tokenizer = MT5Tokenizer.from_pretrained('google/mt5-base')

    # Encode the input text
    inputs = tokenizer.encode('translate English to Italian: ' + color_name, return_tensors='pt')

    # Generate the translated color name
    outputs = model.generate(inputs, max_length=40, num_return_sequences=1)

    # Decode the output back to understandable text
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return decoded_output
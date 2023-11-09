from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# Function to translate English text to French using the google/mt5-base model
# from the Hugging Face Transformers library.
def translate_english_to_french(english_contract_text):
    # Load the pre-trained model and tokenizer
    model = MT5ForConditionalGeneration.from_pretrained('google/mt5-base')
    tokenizer = MT5Tokenizer.from_pretrained('google/mt5-base')

    # Encode the English contract text
    inputs = tokenizer.encode('translate English to French: ' + english_contract_text, return_tensors='pt')

    # Generate the translated text
    outputs = model.generate(inputs, max_length=1000, num_return_sequences=1)

    # Decode the output to obtain the translated French text
    translated_french_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return translated_french_text
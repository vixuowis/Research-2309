from transformers import MarianMTModel, MarianTokenizer

# Function to translate Catalan text to Spanish
# Uses the MarianMTModel and MarianTokenizer from the Hugging Face Transformers library
# The model 'Helsinki-NLP/opus-mt-ca-es' is specifically trained for translation between Catalan and Spanish languages

def translate_catalan_to_spanish(catalan_text):
    # Load the pre-trained model
    model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ca-es')
    # Load the corresponding tokenizer
    tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ca-es')
    # Tokenize the input Catalan text
    tokenized_text = tokenizer.encode(catalan_text, return_tensors="pt")
    # Pass the tokenized text through the model to generate the translated tokens
    translated_tokens = model.generate(tokenized_text)
    # Decode the translated tokens to get the translated text
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text
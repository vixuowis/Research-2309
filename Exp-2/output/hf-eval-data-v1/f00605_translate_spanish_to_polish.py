from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Function to translate Spanish text to Polish
# Uses the Hugging Face's transformers library
# Specifically, the MBartForConditionalGeneration model is used, which is a multilingual machine translation model
# The model has been trained to translate between any pair of 50 languages
# The function takes in Spanish text as input and returns the translated Polish text

def translate_spanish_to_polish(spanish_text):
    # Load the pre-trained model
    model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
    # Load the tokenizer
    tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
    # Set the source language to Spanish
    tokenizer.src_lang = 'es_ES'
    # Tokenize the Spanish text
    encoded_spanish = tokenizer(spanish_text, return_tensors='pt')
    # Generate the translated text
    generated_tokens = model.generate(**encoded_spanish, forced_bos_token_id=tokenizer.lang_code_to_id['pl_PL'])
    # Decode the generated tokens into a human-readable string
    polish_subtitles = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return polish_subtitles
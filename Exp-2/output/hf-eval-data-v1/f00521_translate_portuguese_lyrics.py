from transformers import MarianMTModel, MarianTokenizer

# Function to translate Portuguese lyrics to English
# Uses the MarianMT model from the Hugging Face Transformers library
# The model is pretrained on a large corpus of Portuguese and English text
# The function takes as input a string of Portuguese lyrics and returns the translated English lyrics

def translate_portuguese_lyrics(lyrics):
    # Define the model name
    model_name = 'Helsinki-NLP/opus-mt-pt-en'
    # Load the tokenizer and the model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    # Prepare the batch for the model
    batch = tokenizer.prepare_seq2seq_batch([lyrics])
    # Generate the translation
    gen = model.generate(**batch)
    # Decode the translation and remove special tokens
    translated_lyrics = tokenizer.batch_decode(gen, skip_special_tokens=True)
    # Return the translated lyrics
    return translated_lyrics[0]
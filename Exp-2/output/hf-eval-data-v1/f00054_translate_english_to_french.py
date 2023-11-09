from transformers import pipeline

# Function to translate English text to French
# Uses the Helsinki-NLP/opus-mt-en-fr model from Hugging Face
# The model is trained on the OPUS dataset
# It uses a transformer-align architecture with normalization and SentencePiece pre-processing

def translate_english_to_french(input_text):
    # Instantiate a translation model using the pipeline function
    # with the 'translation_en_to_fr' task
    translate = pipeline('translation_en_to_fr', model='Helsinki-NLP/opus-mt-en-fr')
    
    # Pass the user's input text to the translate function
    translated_text = translate(input_text)
    
    # The function returns the translated text in French
    return translated_text[0]['translation_text']
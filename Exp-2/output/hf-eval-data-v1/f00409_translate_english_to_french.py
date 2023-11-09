from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

# Function to translate English text to French using the T5 model
# @param: input_text - The English text to be translated
# @return: translated_text - The translated French text

def translate_english_to_french(input_text):
    # Load the pre-trained T5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained('google/byt5-small')
    tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

    # Tokenize the input text and convert it into model_inputs
    input_ids = tokenizer.encode(f"translate English to French: {input_text}", return_tensors="pt")

    # Pass the model_inputs to the T5 model, and it will generate a French translation of the article
    output_ids = model.generate(input_ids)

    # Decode the output_ids to get the translated text
    translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return translated_text
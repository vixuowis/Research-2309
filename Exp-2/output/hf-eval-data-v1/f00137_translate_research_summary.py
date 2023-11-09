from transformers import T5Tokenizer, T5Model

# Function to translate research summary from English to Chinese
# using the T5 small model from Hugging Face Transformers

def translate_research_summary(research_summary):
    # Create an instance of the tokenizer and the model
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5Model.from_pretrained('t5-small')

    # Prepare the input text and encode it using the tokenizer
    input_text = f"translate English to Chinese: {research_summary}"
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids

    # Pass the input_ids to the model to generate the translation in Chinese
    decoded_text = model.generate(input_ids)

    # Decode the translated text
    translated_summary = tokenizer.batch_decode(decoded_text, skip_special_tokens=True)

    return translated_summary
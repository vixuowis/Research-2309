from transformers import T5Tokenizer, T5ForConditionalGeneration

def translate_english_to_french(input_text: str) -> str:
    """
    Translates English text to French using the T5-3B model from Hugging Face Transformers.

    Args:
        input_text (str): The English text to be translated.

    Returns:
        str: The translated French text.
    """
    tokenizer = T5Tokenizer.from_pretrained('t5-3b')
    model = T5ForConditionalGeneration.from_pretrained('t5-3b')
    inputs = tokenizer.encode('translate English to French: ' + input_text, return_tensors='pt')
    outputs = model.generate(inputs)
    return tokenizer.decode(outputs[0])
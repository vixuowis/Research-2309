from transformers import AutoModel, AutoTokenizer

def translate_swedish_to_english(input_text: str) -> str:
    """
    Translate Swedish text to English using the Helsinki-NLP/opus-mt-sv-en model from Hugging Face Transformers.

    Args:
        input_text (str): The Swedish text to be translated.

    Returns:
        str: The translated English text.
    """
    model = AutoModel.from_pretrained('Helsinki-NLP/opus-mt-sv-en')
    tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-sv-en')

    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return translated_text
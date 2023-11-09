from transformers import T5Tokenizer, T5ForConditionalGeneration

def translate_and_summarize(input_text: str) -> str:
    """
    This function uses the FLAN-T5 large model from Hugging Face Transformers to translate and summarize text.
    The model is fine-tuned on over 1000 tasks and multiple languages and achieves state-of-the-art performance on several benchmarks.
    It can be used for research on language models, zero-shot NLP tasks, in-context few-shot learning NLP tasks, reasoning, question answering, and advancing fairness and safety research.

    Args:
        input_text (str): The text to be translated and summarized.

    Returns:
        str: The translated and summarized text.
    """
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0])
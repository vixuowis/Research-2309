from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


def multilingual_ner(text):
    """
    This function uses a pre-trained Named Entity Recognition (NER) model from Hugging Face Transformers
    to extract named entities from a given text. The model supports 9 languages (de, en, es, fr, it, nl, pl, pt, ru).
    
    Args:
    text (str): The text from which to extract named entities.
    
    Returns:
    list: A list of dictionaries, each containing information about a named entity found in the text.
    """
    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('Babelscape/wikineural-multilingual-ner')
    model = AutoModelForTokenClassification.from_pretrained('Babelscape/wikineural-multilingual-ner')
    
    # Create a NER pipeline
    nlp = pipeline('ner', model=model, tokenizer=tokenizer)
    
    # Use the pipeline to extract named entities from the text
    ner_results = nlp(text)
    
    return ner_results
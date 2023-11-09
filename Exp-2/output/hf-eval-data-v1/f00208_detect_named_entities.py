from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def detect_named_entities(text):
    """
    This function uses a pre-trained model from the transformers library to detect named entities in a given text.
    The model is capable of recognizing entities in multiple languages.
    
    Parameters:
    text (str): The text in which to detect named entities.
    
    Returns:
    list: A list of dictionaries, each containing information about a detected entity.
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')
    model = AutoModelForTokenClassification.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')
    
    # Create a pipeline for named entity recognition
    nlp = pipeline('ner', model=model, tokenizer=tokenizer)
    
    # Use the pipeline to detect entities in the text
    ner_results = nlp(text)
    
    return ner_results
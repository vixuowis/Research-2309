from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


def extract_locations(text):
    """
    This function takes a multilingual text as input and returns the named entities (locations) in the text.
    It uses the 'Babelscape/wikineural-multilingual-ner' model from Hugging Face Transformers for Named Entity Recognition.
    
    Args:
    text (str): The multilingual text from which to extract locations.
    
    Returns:
    list: A list of named entities (locations) in the text.
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('Babelscape/wikineural-multilingual-ner')
    model = AutoModelForTokenClassification.from_pretrained('Babelscape/wikineural-multilingual-ner')
    
    # Create an NER pipeline
    nlp = pipeline('ner', model=model, tokenizer=tokenizer)
    
    # Process the text and extract the named entities
    ner_results = nlp(text)
    
    # Filter the results to get only the locations
    locations = [result['word'] for result in ner_results if result['entity'] == 'LOC']
    
    return locations
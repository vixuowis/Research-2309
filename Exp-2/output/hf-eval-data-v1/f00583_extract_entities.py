from transformers import pipeline


def extract_entities(text):
    """
    This function extracts the names of companies and people mentioned in the text using Named Entity Recognition.
    
    Parameters:
    text (str): The text from which to extract entities.
    
    Returns:
    list: A list of entities (people and companies) mentioned in the text.
    """
    # Create an NER model using the pipeline function from transformers
    ner_model = pipeline('ner', model='Jean-Baptiste/roberta-large-ner-english')
    
    # Use the created NER model to process the given text
    ner_results = ner_model(text)
    
    # Extract the tokens marked as 'PER' or 'ORG' to get the names of people and companies
    entities = [result['word'] for result in ner_results if result['entity'] in ['PER', 'ORG']]
    
    return entities
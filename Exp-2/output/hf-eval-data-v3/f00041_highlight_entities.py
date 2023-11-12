# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def highlight_entities(text):
    '''
    This function highlights the names of organizations or cities within a given text.
    
    Args:
        text (str): The input text in French.
    
    Returns:
        str: The input text with names of organizations or cities highlighted.
    
    Raises:
        ValueError: If the input is not a string.
    '''
    if not isinstance(text, str):
        raise ValueError('Input text must be a string.')
    
    tokenizer = AutoTokenizer.from_pretrained('Jean-Baptiste/camembert-ner')
    model = AutoModelForTokenClassification.from_pretrained('Jean-Baptiste/camembert-ner')
    nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy='simple')
    
    entities = nlp(text)
    
    highlighted_text = []
    previous_offset = 0
    for entity in entities:
        start_offset, end_offset = entity['start'], entity['end']
        label = entity['entity']
        if label in ['B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']:
            highlighted_text.append(text[previous_offset:start_offset])
            highlighted_text.append(f'[{text[start_offset:end_offset]}]')
            previous_offset = end_offset
    highlighted_text.append(text[previous_offset:])
    
    return ''.join(highlighted_text)

# test_function_code --------------------

def test_highlight_entities():
    '''
    This function tests the highlight_entities function.
    '''
    assert highlight_entities('La société de Paris est spécialisée dans la vente de véhicules électriques.') == 'La société de [Paris] est spécialisée dans la vente de véhicules électriques.'
    assert highlight_entities('Responsable des ventes, vous travaillerez au sein d'une équipe dynamique dans l'agence de Lyon.') == 'Responsable des ventes, vous travaillerez au sein d'une équipe dynamique dans l'agence de [Lyon].'
    assert highlight_entities('Une expérience préalable chez Renault est un atout.') == 'Une expérience préalable chez [Renault] est un atout.'
    assert highlight_entities('') == ''
    assert highlight_entities('Il n'y a pas de noms d'organisations ou de villes dans ce texte.') == 'Il n'y a pas de noms d'organisations ou de villes dans ce texte.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_highlight_entities()
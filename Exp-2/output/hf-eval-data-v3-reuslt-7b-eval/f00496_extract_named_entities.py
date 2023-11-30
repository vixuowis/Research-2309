# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def extract_named_entities(text):
    """
    Extract named entities from a given text using a pre-trained NER model.

    Args:
        text (str): The text from which to extract named entities.

    Returns:
        list: A list of dictionaries. Each dictionary represents a named entity and contains the entity, its start and end indices in the text, and its entity type.
    """
    
    # Tokenize the input text.
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    tokens = tokenizer(text, return_tensors="pt")        
    
    # Perform NER using a pre-trained model.
    nlp = pipeline('ner', model='dslim/bert-base-NER', tokenizer=tokenizer)
    ner_results = nlp(text)
    
    # Extract the results into a list of dictionaries.
    entities = []

    for entity in ner_results:
        start, end = entity["start"], entity["end"]
        
        # The tokenizer returns tokens that are separated with spaces by default (i.e. '##' or 'Ġ'). 
        # We use this to our advantage so that we can calculate indices for entities with whitespace(s).
        entity_tokens = [t[2:] if t.startswith('##') or t.startswith("Ġ") else " "+t for t in tokens['input_ids'][0][start:end+1]]
        
        # Remove the whitespace token that was added to calculate indices with whitespace(s).
        entity_tokens = [et for et in entity_tokens if et != " "]
        
        entities.append({"entity": ' '.join(entity_tokens), 
                         "start": start, 
                         "end": end, 
                         "type": entity["label"]})
    
    # Return the list of dictionaries.
    return entities

# test_function_code --------------------

def test_extract_named_entities():
    """
    Test the function extract_named_entities.
    """
    # Test case 1: English text
    text1 = 'My name is Wolfgang and I live in Berlin.'
    result1 = extract_named_entities(text1)
    assert isinstance(result1, list) and isinstance(result1[0], dict)

    # Test case 2: German text
    text2 = 'Ich heiße Wolfgang und ich wohne in Berlin.'
    result2 = extract_named_entities(text2)
    assert isinstance(result2, list) and isinstance(result2[0], dict)

    # Test case 3: Spanish text
    text3 = 'Mi nombre es Wolfgang y vivo en Berlín.'
    result3 = extract_named_entities(text3)
    assert isinstance(result3, list) and isinstance(result3[0], dict)

    return 'All Tests Passed'


# call_test_function_code --------------------

print(test_extract_named_entities())
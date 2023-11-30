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

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    # create a NER pipeline
    entities = pipeline('ner', model=model, tokenizer=tokenizer)

    # extract named entities from the text using the pipeline
    results = entities(text)
    
    # save all found entities in a list of dictionaries (each dictionary contains an entity)
    entity_list = []

    for i in range(len(results)):
        entity = {}
        
        entity['entity'] = results[i]['word']
        entity['start_index'] = results[i]['start']
        entity['end_index'] = results[i]['end'] 
        entity['type'] = results[i]['entity_group']
        
        entity_list.append(entity)
    
    return(entity_list)

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
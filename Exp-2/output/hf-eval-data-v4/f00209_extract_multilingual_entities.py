# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def extract_multilingual_entities(text):
    """
    Extracts named entities from multilingual text using a pre-trained NER model.
    
    Parameters:
        text (str): Multilingual text from which to extract named entities.
    
    Returns:
        list: A list of named entities extracted from the text.
    """
    tokenizer = AutoTokenizer.from_pretrained('Babelscape/wikineural-multilingual-ner')
    model = AutoModelForTokenClassification.from_pretrained('Babelscape/wikineural-multilingual-ner')
    ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer)
    ner_results = ner_pipeline(text)
    return ner_results

# test_function_code --------------------

def test_extract_multilingual_entities():
    print("Testing started.")
    
    # Test case 1: Multilingual entity extraction
    print("Testing case [1/1] started.")
    example_text = 'My name is Wolfgang and I live in Berlin. Mi nombre es Jos\u00e9 y vivo en Madrid.'
    entities = extract_multilingual_entities(example_text)
    assert len(entities) > 0, f"Test case [1/1] failed: No entities extracted"
    print("Entities extracted: ", entities)
    print("Testing finished.")

# Run the test function
test_extract_multilingual_entities()
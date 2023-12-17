# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def extract_entities(text):
    """
    Extract named entities (people, organizations, and locations) from text using a multilingual NER model.
    
    Args:
        text (str): The news article or text from which to extract entities.
    
    Returns:
        list: A list of entities recognized in the text tagged with their corresponding type (person, organization, location).
    
    Raises:
        ValueError: If 'text' is not a string or is empty.
    """
    if not isinstance(text, str) or not text:
        raise ValueError("The text argument must be a non-empty string.")

    tokenizer = AutoTokenizer.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')
    model = AutoModelForTokenClassification.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')
    nlp = pipeline('ner', model=model, tokenizer=tokenizer)
    entities = nlp(text)
    return entities

# test_function_code --------------------

def test_extract_entities():
    print("Testing started.")

    # Test case 1: English sample
    english_text = "John Doe works at Google headquarters in Mountain View, California."
    print("Testing case [1/3] started.")
    english_results = extract_entities(english_text)
    assert english_results and isinstance(english_results, list), "Test case [1/3] failed: Expected a list of entities."

    # Test case 2: French sample
    french_text = "Emmanuel Macron est le président de la France."
    print("Testing case [2/3] started.")
    french_results = extract_entities(french_text)
    assert french_results and isinstance(french_results, list), "Test case [2/3] failed: Expected a list of entities."

    # Test case 3: Chinese sample
    chinese_text = "习近平是中国的国家主席。"
    print("Testing case [3/3] started.")
    chinese_results = extract_entities(chinese_text)
    assert chinese_results and isinstance(chinese_results, list), "Test case [3/3] failed: Expected a list of entities."

    print("Testing finished.")

# call_test_function_line --------------------

test_extract_entities()
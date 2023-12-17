# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def print_named_entities(text):
    """
    Identify and print named entities in the provided text using a pre-trained NER model.
    
    Args:
        text (str): The input text where named entities are to be identified.
    """
    tokenizer = AutoTokenizer.from_pretrained('Babelscape/wikineural-multilingual-ner')
    model = AutoModelForTokenClassification.from_pretrained('Babelscape/wikineural-multilingual-ner')
    nlp = pipeline('ner', model=model, tokenizer=tokenizer)
    ner_results = nlp(text)
    
    for entity in ner_results:
        print(f"Entity: {entity['word']}, Label: {entity['entity_group']}")

# test_function_code --------------------

def test_print_named_entities():
    print("Testing started.")
    
    # 测试用例 1：英文文本
    print("Testing case [1/3] started.")
    example_text_en = "Alice lives in Zurich and works for the United Nations."
    try:
        print_named_entities(example_text_en)
        print("Test case [1/3] passed.")
    except Exception as e:
        print(f"Test case [1/3] failed: {e}")

    # 测试用例 2：西班牙文文本
    print("Testing case [2/3] started.")
    example_text_es = "Alicia vive en Zúrich y trabaja para las Naciones Unidas."
    try:
        print_named_entities(example_text_es)
        print("Test case [2/3] passed.")
    except Exception as e:
        print(f"Test case [2/3] failed: {e}")

    # 测试用例 3：德文文本
    print("Testing case [3/3] started.")
    example_text_de = "Alice lebt in Zürich und arbeitet für die Vereinten Nationen."
    try:
        print_named_entities(example_text_de)
        print("Test case [3/3] passed.")
    except Exception as e:
        print(f"Test case [3/3] failed: {e}")

    print("Testing finished.")
# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def extract_food_entities(text: str) -> list:
    """Extracts food-related entities from a given text using a NER model.

    Args:
        text (str): A string containing the user input.

    Returns:
        list: A list of dictionaries where each dictionary represents a detected food entity
              along with its position in the text.

    Raises:
        ValueError: If the input text is empty or None.

    """
    if not text:
        raise ValueError('Input text cannot be empty.')

    # Load the pretrained tokenizer and model for food entity recognition
    tokenizer = AutoTokenizer.from_pretrained('Dizex/InstaFoodRoBERTa-NER')
    model = AutoModelForTokenClassification.from_pretrained('Dizex/InstaFoodRoBERTa-NER')

    # Create a pipeline for NER
    ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy='simple')

    # Extract entities
    entities = ner_pipeline(text)
    food_entities = [entity for entity in entities if entity['entity_group'] == 'FOOD']

    return food_entities

# test_function_code --------------------

def test_extract_food_entities():
    print("Testing started.")
    # Sample data for testing
    sample_text = "I had a delicious pepperoni pizza for lunch today!"

    # Testing cases
    print("Testing case [1/1] started.")
    result = extract_food_entities(sample_text)
    assert len(result) > 0, f"Test case [1/1] failed: No entities extracted."
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_food_entities()
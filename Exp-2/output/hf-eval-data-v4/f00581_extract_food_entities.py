# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def extract_food_entities(text):
    """
    Extracts food entities from a given text using a pre-trained NER model.

    Parameters:
        text (str): The text from which to extract food entities.

    Returns:
        list: A list of extracted food entities from the input text.
    """
    tokenizer = AutoTokenizer.from_pretrained('Dizex/InstaFoodRoBERTa-NER')
    model = AutoModelForTokenClassification.from_pretrained('Dizex/InstaFoodRoBERTa-NER')
    ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy='simple')
    food_entities = ner_pipeline(text)
    return [entity['word'] for entity in food_entities if entity['entity_group'] == 'FOOD']

# test_function_code --------------------

def test_extract_food_entities():
    print("Testing extract_food_entities function.")
    test_text = "I loved the sushi and the ramen was exquisite!"
    expected_entities = ['sushi', 'ramen']

    print("Testing with sample text...")
    actual_entities = extract_food_entities(test_text)
    assert set(expected_entities) == set(actual_entities), f"Test failed: expected {expected_entities}, got {actual_entities}"
    print("Test passed successfully.")

# Run the test function
test_extract_food_entities()
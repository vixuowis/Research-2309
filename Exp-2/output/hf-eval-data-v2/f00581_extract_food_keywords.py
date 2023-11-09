# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def extract_food_keywords(user_input):
    """
    This function extracts food-related keywords from the user's input text.

    Args:
        user_input (str): The user's input text.

    Returns:
        food_keywords (list): A list of food-related keywords.
    """
    tokenizer = AutoTokenizer.from_pretrained('Dizex/InstaFoodRoBERTa-NER')
    model = AutoModelForTokenClassification.from_pretrained('Dizex/InstaFoodRoBERTa-NER')
    food_entity_recognition = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy='simple')
    food_keywords = food_entity_recognition(user_input)
    return food_keywords

# test_function_code --------------------

def test_extract_food_keywords():
    """
    This function tests the extract_food_keywords function.
    It uses a sample input and checks if the output is as expected.
    """
    sample_input = "Today's meal: Fresh olive poke bowl topped with chia seeds. Very delicious!"
    expected_output = [{'entity': 'FOOD', 'score': 0.999, 'index': 4, 'start': 18, 'end': 23, 'is_subword': False}, {'entity': 'FOOD', 'score': 0.999, 'index': 6, 'start': 24, 'end': 28, 'is_subword': False}, {'entity': 'FOOD', 'score': 0.999, 'index': 8, 'start': 29, 'end': 33, 'is_subword': False}, {'entity': 'FOOD', 'score': 0.999, 'index': 10, 'start': 34, 'end': 38, 'is_subword': False}, {'entity': 'FOOD', 'score': 0.999, 'index': 12, 'start': 39, 'end': 43, 'is_subword': False}]
    assert len(extract_food_keywords(sample_input)) == len(expected_output)

# call_test_function_code --------------------

test_extract_food_keywords()
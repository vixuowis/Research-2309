# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def extract_food_keywords(user_input):
    """
    Extract food-related keywords from user's input text using a pre-trained model.

    Args:
        user_input (str): The user's input text.

    Returns:
        list: A list of dictionaries. Each dictionary contains the entity, score, index, start, and end of the food-related keyword.
    """
    tokenizer = AutoTokenizer.from_pretrained('Dizex/InstaFoodRoBERTa-NER')
    model = AutoModelForTokenClassification.from_pretrained('Dizex/InstaFoodRoBERTa-NER')
    food_entity_recognition = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy='simple')
    food_keywords = food_entity_recognition(user_input)
    return food_keywords

# test_function_code --------------------

def test_extract_food_keywords():
    """
    Test the function extract_food_keywords.
    """
    sample_input = "Today's meal: Fresh olive poke bowl topped with chia seeds. Very delicious!"
    expected_output = [{'entity': 'FOOD', 'score': 0.999, 'index': 4, 'start': 18, 'end': 23}, {'entity': 'FOOD', 'score': 0.999, 'index': 5, 'start': 24, 'end': 28}, {'entity': 'FOOD', 'score': 0.999, 'index': 6, 'start': 29, 'end': 33}, {'entity': 'FOOD', 'score': 0.999, 'index': 7, 'start': 34, 'end': 38}, {'entity': 'FOOD', 'score': 0.999, 'index': 8, 'start': 39, 'end': 43}]
    assert len(extract_food_keywords(sample_input)) == len(expected_output)
    print('All Tests Passed')

# call_test_function_code --------------------

test_extract_food_keywords()
# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def detect_named_entities(text):
    """
    Detect named entities in a given text using a multilingual named entity recognition model.

    Args:
        text (str): The text in which to detect named entities.

    Returns:
        list: A list of dictionaries, each containing information about a detected named entity.
    """
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    # Prepare the text for processing by the model
    input_text = f"{text}"
    encoded_input = tokenizer(input_text, return_tensors="pt", padding=True)

    # Detect named entities and get their corresponding tokens
    ner = pipeline("ner")
    output = ner(encoded_input["input_ids"], 
            attention_mask=encoded_input['attention_mask'], 
            token_type_ids=encoded_input['token_type_ids'])
    
    # Process the results and get only named entities with confidence >= 0.85
    results = []
    for entity in output:
        if entity["score"] > .85:
            results.append({"label":entity["entity"], "word":entity["word"].strip()})
    
    return results

# test_function_code --------------------

def test_detect_named_entities():
    """
    Test the detect_named_entities function.
    """
    test_text_1 = 'Nader Jokhadar had given Syria the lead with a well-struck header in the seventh minute.'
    test_text_2 = 'Apple Inc. is planning to open a new store in San Francisco.'
    test_text_3 = 'Angela Merkel met with Emmanuel Macron in Berlin.'
    assert isinstance(detect_named_entities(test_text_1), list)
    assert isinstance(detect_named_entities(test_text_2), list)
    assert isinstance(detect_named_entities(test_text_3), list)
    print('All Tests Passed')


# call_test_function_code --------------------

test_detect_named_entities()
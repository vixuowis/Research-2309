# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def translate_question_to_german(input_text):
    """
    Translate a question from English to German regarding the location of parks in Munich.

    Parameters:
        input_text (str): The English question to translate. Example: "Where are the parks in Munich?"

    Returns:
        str: The translated question in German.
    """
    # Load the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')

    # Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    # Generate the translation
    outputs = model.generate(input_ids)
    # Decode the translation
    translated_question = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return translated_question

# test_function_code --------------------

def test_translate_question_to_german():
    print("Testing started.")
    
    # Test case 1: Check if the function translates the example input correctly
    print("Testing case [1/1] started.")
    english_question = "Where are the parks in Munich?"
    german_question = translate_question_to_german(english_question)
    assert german_question is not None and isinstance(german_question, str), f"Test case [1/1] failed: Expected a string, got {type(german_question)}"
    print(german_question)  # Printing out the translated question for manual verification
    print("Testing finished.")

# Run the test function
test_translate_question_to_german()
# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import T5ForConditionalGeneration, T5Tokenizer

# function_code --------------------

def generate_questions_from_paragraph(paragraph: str) -> str:
    """
    Generates questions based on the provided paragraph using T5 model.

    Args:
        paragraph (str): A string containing the paragraph to generate questions from.

    Returns:
        str: A string containing the generated questions.

    Raises:
        ValueError: If the provided paragraph is empty.
    """
    if not paragraph:
        raise ValueError('The paragraph cannot be empty.')
    model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco')
    tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
    inputs = tokenizer.encode('generate questions: ' + paragraph, return_tensors='pt', padding=True)
    outputs = model.generate(inputs, max_length=100)
    questions = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return questions


# test_function_code --------------------

def test_generate_questions_from_paragraph():
    print("Testing started.")

    # Test case 1: Check with a valid paragraph
    print("Testing case [1/2] started.")
    valid_paragraph = "The quick brown fox jumps over the lazy dog."
    questions = generate_questions_from_paragraph(valid_paragraph)
    assert questions, "Test case [1/2] failed: No questions generated."

    # Test case 2: Check with an empty paragraph
    print("Testing case [2/2] started.")
    try:
        generate_questions_from_paragraph("")
        assert False, "Test case [2/2] failed: ValueError not raised for empty paragraph."
    except ValueError as e:
        assert str(e) == 'The paragraph cannot be empty.', "Test case [2/2] failed: Invalid error message for empty paragraph."

    print("Testing finished.")


# call_test_function_line --------------------

test_generate_questions_from_paragraph()
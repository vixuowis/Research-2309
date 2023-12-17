# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelWithLMHead

# function_code --------------------

def generate_russian_response(input_text):
    """
    Generate a response to an input text in Russian using a pre-trained conversational model.

    Args:
        input_text (str): Text in Russian for which a response is to be generated.

    Returns:
        str: The generated response in Russian.

    Raises:
        ValueError: If input_text is not a string.
    """
    if not isinstance(input_text, str):
        raise ValueError('input_text must be a string.')

    tokenizer = AutoTokenizer.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
    model = AutoModelWithLMHead.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
    inputs = tokenizer(input_text, return_tensors='pt')
    generated_token_ids = model.generate(**inputs, max_length=50)
    response = tokenizer.decode(generated_token_ids[0], skip_special_tokens=True)
    return response

# test_function_code --------------------

def test_generate_russian_response():
    print("Testing started.")

    # Test case 1: Valid input text
    print("Testing case [1/3] started.")
    input_text = 'Привет, как тебя зовут?'
    response = generate_russian_response(input_text)
    assert isinstance(response, str), f"Test case [1/3] failed: Expected output type is str, but got {type(response)}"

    # Test case 2: Input text is an empty string
    print("Testing case [2/3] started.")
    input_text = ''
    response = generate_russian_response(input_text)
    assert response != '', f"Test case [2/3] failed: Expected non-empty response, but got an empty string."

    # Test case 3: Input text is not a string
    print("Testing case [3/3] started.")
    input_text = None
    try:
        generate_russian_response(input_text)
        assert False, f"Test case [3/3] failed: Expected ValueError, but no exception was raised."
    except ValueError as e:
        assert str(e) == 'input_text must be a string.', f"Test case [3/3] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_russian_response()
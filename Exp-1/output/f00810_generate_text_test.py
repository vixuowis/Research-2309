from f00810_generate_text import *
def test_generate_text():
    input_text = 'Once upon a time'
    model_name = 'gpt2'
    max_length = 50

    # Generate text
    generated_text = generate_text(input_text, model_name, max_length)

    # Assert generated text is not empty
    assert generated_text != ''

    # Assert generated text is a string
    assert isinstance(generated_text, str)

    # Assert generated text is not equal to input text
    assert generated_text != input_text

    print('All tests passed!')

test_generate_text()

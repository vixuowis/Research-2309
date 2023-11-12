# function_import --------------------

from transformers import TextGenerationPipeline, Bloom7b1Model

# function_code --------------------

def generate_text(prompt):
    """
    Generate a paragraph of text based on a given prompt using the Bloom7b1Model from Hugging Face Transformers.

    Args:
        prompt (str): The initial text to base the generated paragraph on.

    Returns:
        str: The generated paragraph of text.
    """
    model = Bloom7b1Model.from_pretrained('bigscience/bloom-7b1')
    text_generator = TextGenerationPipeline(model=model)
    generated_paragraph = text_generator(prompt)[0]['generated_text']
    return generated_paragraph

# test_function_code --------------------

def test_generate_text():
    """
    Test the generate_text function with various prompts.
    """
    assert generate_text('Once upon a time')
    assert generate_text('In a galaxy far, far away')
    assert generate_text('It was a dark and stormy night')
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_text()
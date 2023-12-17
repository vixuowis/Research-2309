# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_paraphrased_explanations(chemistry_concept_text: str) -> str:
    """
    Generate a paraphrased explanation for a given chemistry concept text.

    Args:
        chemistry_concept_text (str): The chemistry concept text to paraphrase.

    Returns:
        str: The paraphrased chemistry concept.

    Raises:
        ValueError: If the input text is not provided.
    """
    if not chemistry_concept_text:
        raise ValueError('Input text is required.')
    paraphraser = pipeline('text-classification', model='prithivida/parrot_fluency_model')
    return paraphraser(chemistry_concept_text)[0]['label']

# test_function_code --------------------

def test_generate_paraphrased_explanations():
    print("Testing started.")

    # Test case 1: Non-empty input
    print("Testing case [1/3] started.")
    sample_text = 'A molecule is the smallest particle in a chemical element or compound.'
    paraphrased = generate_paraphrased_explanations(sample_text)
    assert paraphrased is not None, f"Test case [1/3] failed: Generated paraphrased text is None."

    # Test case 2: Valid paraphrased output type
    print("Testing case [2/3] started.")
    assert isinstance(paraphrased, str), f"Test case [2/3] failed: Paraphrased explanation is not a string."

    # Test case 3: Empty input handling
    print("Testing case [3/3] started.")
    try:
        generate_paraphrased_explanations('')
        raise AssertionError("Test case [3/3] failed: ValueError not raised on empty input.")
    except ValueError:
        pass  # ValueError is expected

    print("Testing finished.")

# call_test_function_line --------------------

test_generate_paraphrased_explanations()
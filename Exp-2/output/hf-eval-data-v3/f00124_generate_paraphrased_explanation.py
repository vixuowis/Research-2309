# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_paraphrased_explanation(chemistry_concept_text):
    """
    Generate a paraphrased explanation for a given chemistry concept.

    Args:
        chemistry_concept_text (str): The text of the chemistry concept to be paraphrased.

    Returns:
        str: The paraphrased explanation of the given chemistry concept.

    Raises:
        ValueError: If the input is not a string.
    """
    if not isinstance(chemistry_concept_text, str):
        raise ValueError('Input must be a string.')
    paraphraser = pipeline('text-classification', model='prithivida/parrot_fluency_model')
    paraphrased_explanation = paraphraser(chemistry_concept_text)
    return paraphrased_explanation

# test_function_code --------------------

def test_generate_paraphrased_explanation():
    """
    Test the function generate_paraphrased_explanation.
    """
    chemistry_concept_text1 = 'The concept of atomic structure is the basic foundation of chemistry.'
    chemistry_concept_text2 = 'Chemical reactions involve changes in the arrangement of atoms.'
    chemistry_concept_text3 = 'The periodic table is a tabular arrangement of chemical elements.'
    assert isinstance(generate_paraphrased_explanation(chemistry_concept_text1), str), 'The output should be a string.'
    assert isinstance(generate_paraphrased_explanation(chemistry_concept_text2), str), 'The output should be a string.'
    assert isinstance(generate_paraphrased_explanation(chemistry_concept_text3), str), 'The output should be a string.'
    print('All Tests Passed')

# call_test_function_code --------------------

test_generate_paraphrased_explanation()
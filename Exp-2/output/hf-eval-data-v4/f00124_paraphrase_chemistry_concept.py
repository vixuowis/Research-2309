# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def paraphrase_chemistry_concept(chemistry_concept_text):
    # This function takes a chemistry concept in text form and uses the Parrot model
    # to generate a paraphrased version of the explanation.
    # Args:
    #     chemistry_concept_text (str): A text string containing the chemistry concept to paraphrase.
    # Returns:
    #     str: A paraphrased version of the chemistry concept.

    # Initialize the paraphrase model using the Parrot fluency model from Hugging Face Transformers
    paraphraser = pipeline('text-classification', model='prithivida/parrot_fluency_model')

    # Generate the paraphrased explanation
    paraphrased_explanation = paraphraser(chemistry_concept_text)

    # Return the first result from the generated paraphrases
    return paraphrased_explanation[0]['paraphrase']

# test_function_code --------------------

def test_paraphrase_chemistry_concept():
    print("Testing started.")

    # Sample chemistry concept text
    chemistry_concept_text = "A chemical reaction involves the conversion of reactants into products."

    # Expected to get a paraphrase of the chemistry concept
    expected_result_type = str

    # Test case: Ensure the paraphrase function returns a string
    print("Testing case [1/1] started.")
    paraphrased_concept = paraphrase_chemistry_concept(chemistry_concept_text)
    assert isinstance(paraphrased_concept, expected_result_type), f"Test case failed: Expected result type {expected_result_type}, got {type(paraphrased_concept).__name__}"
    print("Test case passed.")
    print("Testing finished.")

# Run the test function
test_paraphrase_chemistry_concept()
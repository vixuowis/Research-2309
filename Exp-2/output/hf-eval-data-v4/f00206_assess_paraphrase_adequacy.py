# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def assess_paraphrase_adequacy(generated_paraphrase):
    """
    Assess the adequacy of a paraphrase generated by an AI chatbot.

    This function utilizes a pretrained model from Hugging Face Transformers
    to classify whether the given paraphrase is adequate or not.

    :param generated_paraphrase: The paraphrased text to be evaluated
    :type generated_paraphrase: str
    :return: The result of adequacy classification
    :rtype: dict
    """
    adequacy_classifier = pipeline('text-classification', model='prithivida/parrot_adequacy_model')
    paraphrase_adequacy = adequacy_classifier(generated_paraphrase)
    return paraphrase_adequacy

# test_function_code --------------------

def test_assess_paraphrase_adequacy():
    print("Testing assess_paraphrase_adequacy function.")

    # Test case 1: Adequate paraphrase
    print("Testing case [1/3] started.")
    adequate_sample = "How can I refund my purchase?"
    result = assess_paraphrase_adequacy(adequate_sample)
    assert 'label' in result[0] and result[0]['label'] == 'ADEQUATE', f"Test case [1/3] failed: {result}"

    # Test case 2: Inadequate paraphrase
    print("Testing case [2/3] started.")
    inadequate_sample = "What's the weather like?"
    result = assess_paraphrase_adequacy(inadequate_sample)
    assert 'label' in result[0] and result[0]['label'] == 'INADEQUATE', f"Test case [2/3] failed: {result}"

    # Test case 3: Unrelated paraphrase
    print("Testing case [3/3] started.")
    unrelated_sample = "I love playing football on weekends."
    result = assess_paraphrase_adequacy(unrelated_sample)
    assert 'label' in result[0] and result[0]['label'] == 'INADEQUATE', f"Test case [3/3] failed: {result}"
    print("Testing finished.")

# Run the test function
test_assess_paraphrase_adequacy()